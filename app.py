import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5 import QtCore, QtGui, QtWidgets


@dataclass
class AppConfig:
    conf: float = 0.25
    iou: float = 0.45
    tracker_key: str = "botsort"
    rider_moto_ioa: float = 0.05
    gear_rider_ioa: float = 0.20


CLASS_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (128, 128, 128),  # Motorcycle - Gray
    1: (255, 0, 0),      # Rider - Blue (BGR)
    2: (0, 255, 0),      # Helmet - Green
    3: (0, 255, 0),      # Footwear - Green
    4: (0, 0, 255),      # Improper_Footwear - Red
}

CLASS_NAMES_DEFAULT: Dict[int, str] = {
    0: "Motorcycle",
    1: "Rider",
    2: "Helmet",
    3: "Footwear",
    4: "Improper_Footwear",
}


TARGET_CLASS_IDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

def _box_area(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _ioa(child: Tuple[int, int, int, int], parent: Tuple[int, int, int, int]) -> float:
    area_child = _box_area(child)
    if area_child <= 0:
        return 0.0
    inter = _intersection_area(child, parent)
    return inter / float(area_child)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def tracker_yaml_path(tracker_key: str) -> str:
    local = os.path.join(repo_root(), "trackers")
    candidates = {
        "botsort": os.path.join(local, "botsort.yaml"),
        "bytetrack": os.path.join(local, "bytetrack.yaml"),
        "strongsort": os.path.join(local, "strongsort.yaml"),
    }
    return candidates.get(tracker_key, candidates["botsort"])


class VideoThread(QtCore.QThread):
    frameReady = QtCore.pyqtSignal(QtGui.QImage)
    statusReady = QtCore.pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._video_path: Optional[str] = None
        self._model_paths: Dict[int, Optional[str]] = {cid: None for cid in TARGET_CLASS_IDS}
        self._config = AppConfig()
        self._stop = False
        self._paused = True
        self._restart_requested = False
        self._models: Dict[int, YOLO] = {}

        self._frame_delay_ms: int = 1

    def set_video_path(self, path: Optional[str]) -> None:
        self._video_path = path

    def set_model_path_for_class(self, class_id: int, path: Optional[str]) -> None:
        if class_id not in self._model_paths:
            return
        self._model_paths[class_id] = path
        self._restart_requested = True

    def set_conf(self, conf: float) -> None:
        self._config.conf = clamp01(conf)

    def set_iou(self, iou: float) -> None:
        self._config.iou = clamp01(iou)

    def set_tracker(self, tracker_key: str) -> None:
        self._config.tracker_key = tracker_key
        self._restart_requested = True

    def set_rider_moto_ioa(self, value: float) -> None:
        self._config.rider_moto_ioa = max(0.0, min(1.0, float(value)))

    def set_gear_rider_ioa(self, value: float) -> None:
        self._config.gear_rider_ioa = max(0.0, min(1.0, float(value)))

    def play(self) -> None:
        self._paused = False

    def pause(self) -> None:
        self._paused = True

    def stop(self) -> None:
        self._stop = True
        self._paused = True

    def _emit_status(self, text: str) -> None:
        self.statusReady.emit(text)

    def _load_model(self) -> None:
        rider_path = self._model_paths.get(1)
        if not rider_path:
            raise RuntimeError("No Rider model selected (class 1)")

        # Load selected models; Rider (class 1) is required for tracking.
        self._models.clear()
        for class_id, path in self._model_paths.items():
            if not path:
                continue
            self._emit_status(
                f"Loading model for {CLASS_NAMES_DEFAULT.get(class_id, str(class_id))}: {os.path.basename(path)}"
            )
            self._models[class_id] = YOLO(path)

        if 1 not in self._models:
            # Should not happen, but keep the error explicit.
            raise RuntimeError("Failed to load Rider model")

    def _collect_dets(
        self,
        result,
        class_id_override: int,
        include_track_ids: bool,
    ) -> List[Tuple[int, int, int, int, int, float, Optional[int]]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = getattr(boxes, "xyxy", None)
        conf = getattr(boxes, "conf", None)
        ids = getattr(boxes, "id", None)

        if xyxy is None:
            return []

        xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)

        conf_arr = None
        if conf is not None:
            conf_arr = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)

        id_arr = None
        if include_track_ids and ids is not None:
            try:
                id_arr = ids.cpu().numpy() if hasattr(ids, "cpu") else np.asarray(ids)
            except Exception:
                id_arr = None

        dets: List[Tuple[int, int, int, int, int, float, Optional[int]]] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
            score = float(conf_arr[i]) if conf_arr is not None else 0.0
            track_id: Optional[int] = None
            if id_arr is not None and i < len(id_arr) and id_arr[i] is not None:
                try:
                    track_id = int(id_arr[i])
                except Exception:
                    track_id = None
            dets.append((x1, y1, x2, y2, int(class_id_override), score, track_id))

        return dets

    def _annotate(self, frame_bgr: np.ndarray, dets) -> np.ndarray:
        annotated = frame_bgr.copy()

        for (x1, y1, x2, y2, class_id, score, track_id) in dets:
            color = CLASS_COLORS_BGR.get(class_id, (0, 255, 255))

            label = CLASS_NAMES_DEFAULT.get(class_id, str(class_id))
            label += f" {score:.2f}"

            if class_id == 1 and track_id is not None:
                label += f" ID:{track_id}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            tx1, ty1 = x1, max(0, y1 - th - 8)
            cv2.rectangle(annotated, (tx1, ty1), (tx1 + tw + 6, ty1 + th + 6), color, -1)
            cv2.putText(
                annotated,
                label,
                (tx1 + 3, ty1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return annotated

    def _filter_dets_by_overlap(self, dets):
        """Reduce false positives by enforcing expected spatial relationships.

        Rules:
        - Rider (1) must overlap a Motorcycle (0)
        - Helmet (2), Footwear (3), Improper_Footwear (4) must overlap a Rider (1)
        """
        motorcycles = [d for d in dets if d[4] == 0]
        riders = [d for d in dets if d[4] == 1]
        gear = [d for d in dets if d[4] in (2, 3, 4)]

        other = [d for d in dets if d[4] not in (0, 1, 2, 3, 4)]

        moto_boxes = [(d[0], d[1], d[2], d[3]) for d in motorcycles]

        # Filter riders by overlap with motorcycles
        kept_riders = []
        rider_boxes = []
        if moto_boxes:
            for d in riders:
                rbox = (d[0], d[1], d[2], d[3])
                ok = any(_ioa(rbox, mbox) >= self._config.rider_moto_ioa for mbox in moto_boxes)
                if ok:
                    kept_riders.append(d)
                    rider_boxes.append(rbox)
        else:
            # If there are no motorcycles detected, keep zero riders to avoid pedestrian false positives.
            kept_riders = []
            rider_boxes = []

        # Filter gear by overlap with kept riders
        kept_gear = []
        if rider_boxes:
            for d in gear:
                gbox = (d[0], d[1], d[2], d[3])
                ok = any(_ioa(gbox, rbox) >= self._config.gear_rider_ioa for rbox in rider_boxes)
                if ok:
                    kept_gear.append(d)

        return motorcycles + kept_riders + kept_gear + other

    def run(self) -> None:
        self._stop = False
        self._paused = True

        if not self._video_path:
            self._emit_status("Select a video to start")
            return

        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            self._emit_status("Failed to open video")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            self._frame_delay_ms = int(1000 / fps)
        else:
            self._frame_delay_ms = 1

        try:
            self._load_model()
        except Exception as e:
            self._emit_status(f"Model load error: {e}")
            cap.release()
            return

        self._emit_status("Ready (press Play)")

        last_time = time.time()
        frame_count = 0

        while not self._stop:
            if self._paused:
                self.msleep(25)
                continue

            if self._restart_requested:
                self._restart_requested = False
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Reload models so model-path changes take effect.
                try:
                    self._load_model()
                except Exception as e:
                    self._emit_status(f"Model load error: {e}")
                    self._paused = True
                    continue
                self._emit_status("Restarted video due to configuration change")

            ok, frame = cap.read()
            if not ok:
                self._emit_status("End of video")
                self._paused = True
                continue

            tracker_path = tracker_yaml_path(self._config.tracker_key)
            if self._config.tracker_key == "strongsort" and not os.path.exists(tracker_path):
                tracker_path = tracker_yaml_path("bytetrack")

            try:
                # 1) Rider model drives tracking.
                rider_model = self._models[1]
                track_results = rider_model.track(
                    frame,
                    conf=self._config.conf,
                    iou=self._config.iou,
                    persist=True,
                    tracker=tracker_path,
                    verbose=False,
                )

                track_result0 = track_results[0] if isinstance(track_results, (list, tuple)) else track_results
                dets = self._collect_dets(track_result0, class_id_override=1, include_track_ids=True)

                # 2) Other single-class models run detection (no tracking IDs).
                for class_id in TARGET_CLASS_IDS:
                    if class_id == 1:
                        continue
                    model = self._models.get(class_id)
                    if model is None:
                        continue
                    pred_results = model.predict(
                        frame,
                        conf=self._config.conf,
                        iou=self._config.iou,
                        verbose=False,
                    )
                    pred0 = pred_results[0] if isinstance(pred_results, (list, tuple)) else pred_results
                    dets.extend(self._collect_dets(pred0, class_id_override=class_id, include_track_ids=False))

                dets = self._filter_dets_by_overlap(dets)
                annotated = self._annotate(frame, dets)
            except Exception as e:
                self._emit_status(f"Inference error: {e}")
                self._paused = True
                continue

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
            self.frameReady.emit(qimg)

            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                self._emit_status(
                    f"Running | conf={self._config.conf:.2f} iou={self._config.iou:.2f} "
                    f"tracker={self._config.tracker_key} | FPS~{frame_count / (now - last_time):.1f}"
                )
                last_time = now
                frame_count = 0

            if self._frame_delay_ms > 1:
                self.msleep(max(1, self._frame_delay_ms // 2))

        cap.release()
        self._emit_status("Stopped")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Safety Gear Compliance Tester (YOLO + Tracking)")

        self._video_path: Optional[str] = None
        self._model_paths: Dict[int, Optional[str]] = {cid: None for cid in TARGET_CLASS_IDS}

        self._thread = VideoThread()
        self._thread.frameReady.connect(self._on_frame)
        self._thread.statusReady.connect(self._set_status)

        self._build_ui()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        self.videoLabel = QtWidgets.QLabel("Load a video and model", self)
        self.videoLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.videoLabel.setMinimumSize(960, 540)
        self.videoLabel.setStyleSheet("background-color: #111; color: #ddd;")

        self.btnLoadVideo = QtWidgets.QPushButton("Load Video (.mp4)")

        self.btnLoadMotorcycle = QtWidgets.QPushButton("Load Motorcycle Model")
        self.btnLoadRider = QtWidgets.QPushButton("Load Rider Model (required)")
        self.btnLoadHelmet = QtWidgets.QPushButton("Load Helmet Model")
        self.btnLoadFootwear = QtWidgets.QPushButton("Load Footwear Model")
        self.btnLoadImproperFootwear = QtWidgets.QPushButton("Load Improper Footwear Model")

        self.lblMotorcycle = QtWidgets.QLabel("(not set)")
        self.lblRider = QtWidgets.QLabel("(not set)")
        self.lblHelmet = QtWidgets.QLabel("(not set)")
        self.lblFootwear = QtWidgets.QLabel("(not set)")
        self.lblImproperFootwear = QtWidgets.QLabel("(not set)")

        self.btnPlay = QtWidgets.QPushButton("Play")
        self.btnPause = QtWidgets.QPushButton("Pause")
        self.btnStop = QtWidgets.QPushButton("Stop")

        self.trackerCombo = QtWidgets.QComboBox()
        self.trackerCombo.addItem("BoT-SORT (primary)", "botsort")
        self.trackerCombo.addItem("StrongSORT (if available)", "strongsort")
        self.trackerCombo.addItem("ByteTrack (comparison)", "bytetrack")

        self.confSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.confSlider.setRange(0, 100)
        self.confSlider.setValue(25)

        self.iouSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.iouSlider.setRange(0, 100)
        self.iouSlider.setValue(45)

        self.riderMotoSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.riderMotoSlider.setRange(0, 100)
        self.riderMotoSlider.setValue(5)

        self.gearRiderSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gearRiderSlider.setRange(0, 100)
        self.gearRiderSlider.setValue(20)

        self.confValue = QtWidgets.QLabel("0.25")
        self.iouValue = QtWidgets.QLabel("0.45")
        self.riderMotoValue = QtWidgets.QLabel("0.05")
        self.gearRiderValue = QtWidgets.QLabel("0.20")

        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.btnLoadVideo, 0, 0, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Tracker"), 0, 2)
        grid.addWidget(self.trackerCombo, 0, 3)

        grid.addWidget(self.btnLoadMotorcycle, 1, 0)
        grid.addWidget(self.lblMotorcycle, 1, 1)
        grid.addWidget(self.btnLoadRider, 1, 2)
        grid.addWidget(self.lblRider, 1, 3)

        grid.addWidget(self.btnLoadHelmet, 2, 0)
        grid.addWidget(self.lblHelmet, 2, 1)
        grid.addWidget(self.btnLoadFootwear, 2, 2)
        grid.addWidget(self.lblFootwear, 2, 3)

        grid.addWidget(self.btnLoadImproperFootwear, 3, 0)
        grid.addWidget(self.lblImproperFootwear, 3, 1, 1, 3)

        grid.addWidget(QtWidgets.QLabel("Conf Threshold"), 4, 0)
        grid.addWidget(self.confSlider, 4, 1, 1, 2)
        grid.addWidget(self.confValue, 4, 3)

        grid.addWidget(QtWidgets.QLabel("IoU Threshold"), 5, 0)
        grid.addWidget(self.iouSlider, 5, 1, 1, 2)
        grid.addWidget(self.iouValue, 5, 3)

        grid.addWidget(QtWidgets.QLabel("Rider requires Motorcycle (IoA)"), 6, 0)
        grid.addWidget(self.riderMotoSlider, 6, 1, 1, 2)
        grid.addWidget(self.riderMotoValue, 6, 3)

        grid.addWidget(QtWidgets.QLabel("Gear requires Rider (IoA)"), 7, 0)
        grid.addWidget(self.gearRiderSlider, 7, 1, 1, 2)
        grid.addWidget(self.gearRiderValue, 7, 3)

        btnRow = QtWidgets.QHBoxLayout()
        btnRow.addWidget(self.btnPlay)
        btnRow.addWidget(self.btnPause)
        btnRow.addWidget(self.btnStop)

        self.statusBar = QtWidgets.QLabel("Idle")

        layout = QtWidgets.QVBoxLayout(central)
        layout.addLayout(grid)
        layout.addLayout(btnRow)
        layout.addWidget(self.videoLabel, stretch=1)
        layout.addWidget(self.statusBar)

        self.btnLoadVideo.clicked.connect(self._pick_video)
        self.btnLoadMotorcycle.clicked.connect(lambda: self._pick_model_for_class(0))
        self.btnLoadRider.clicked.connect(lambda: self._pick_model_for_class(1))
        self.btnLoadHelmet.clicked.connect(lambda: self._pick_model_for_class(2))
        self.btnLoadFootwear.clicked.connect(lambda: self._pick_model_for_class(3))
        self.btnLoadImproperFootwear.clicked.connect(lambda: self._pick_model_for_class(4))

        self.btnPlay.clicked.connect(self._play)
        self.btnPause.clicked.connect(self._pause)
        self.btnStop.clicked.connect(self._stop)

        self.confSlider.valueChanged.connect(self._on_conf_changed)
        self.iouSlider.valueChanged.connect(self._on_iou_changed)
        self.riderMotoSlider.valueChanged.connect(self._on_rider_moto_changed)
        self.gearRiderSlider.valueChanged.connect(self._on_gear_rider_changed)
        self.trackerCombo.currentIndexChanged.connect(self._on_tracker_changed)

    def _set_status(self, text: str) -> None:
        self.statusBar.setText(text)

    def _pick_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        if not path:
            return
        self._video_path = path
        self._thread.set_video_path(path)
        self._set_status(f"Video selected: {os.path.basename(path)}")

        # Quick sanity preview to confirm OpenCV can open/decode this file.
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(
                self,
                "Video Open Failed",
                "OpenCV could not open the selected video.\n\n"
                "Common causes:\n"
                "- Unsupported codec (e.g., some H.265/HEVC MP4s)\n"
                "- File path permissions\n\n"
                f"Path:\n{path}",
            )
            self._set_status("Failed to open selected video")
            return

        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            QtWidgets.QMessageBox.critical(
                self,
                "Video Decode Failed",
                "OpenCV opened the file but could not decode the first frame.\n\n"
                "Try re-encoding the video to H.264 (AVC) or using a different file.",
            )
            self._set_status("Failed to decode video")
            return

        # Display preview frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
        self._on_frame(qimg)

    def _pick_model_for_class(self, class_id: int) -> None:
        title = f"Select {CLASS_NAMES_DEFAULT.get(class_id, str(class_id))} Model"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, "", "PyTorch Model (*.pt);;All Files (*)")
        if not path:
            return

        self._model_paths[class_id] = path
        self._thread.set_model_path_for_class(class_id, path)

        base = os.path.basename(path)
        if class_id == 0:
            self.lblMotorcycle.setText(base)
        elif class_id == 1:
            self.lblRider.setText(base)
        elif class_id == 2:
            self.lblHelmet.setText(base)
        elif class_id == 3:
            self.lblFootwear.setText(base)
        elif class_id == 4:
            self.lblImproperFootwear.setText(base)

        self._set_status(f"Model set for {CLASS_NAMES_DEFAULT.get(class_id, str(class_id))}: {base}")

    def _ensure_thread_started(self) -> None:
        if self._thread.isRunning():
            return
        self._thread.start()

    def _play(self) -> None:
        if not self._video_path:
            self._set_status("Select a video first")
            return
        if not self._model_paths.get(1):
            self._set_status("Select the Rider model (required) to enable tracking")
            return
        self._ensure_thread_started()
        self._thread.play()
        self._set_status("Playing")

    def _pause(self) -> None:
        self._thread.pause()
        self._set_status("Paused")

    def _stop(self) -> None:
        if self._thread.isRunning():
            self._thread.stop()
            self._thread.wait(1500)
        self._set_status("Stopped")

    def _on_conf_changed(self, value: int) -> None:
        conf = value / 100.0
        self.confValue.setText(f"{conf:.2f}")
        self._thread.set_conf(conf)

    def _on_iou_changed(self, value: int) -> None:
        iou = value / 100.0
        self.iouValue.setText(f"{iou:.2f}")
        self._thread.set_iou(iou)

    def _on_rider_moto_changed(self, value: int) -> None:
        thr = value / 100.0
        self.riderMotoValue.setText(f"{thr:.2f}")
        self._thread.set_rider_moto_ioa(thr)

    def _on_gear_rider_changed(self, value: int) -> None:
        thr = value / 100.0
        self.gearRiderValue.setText(f"{thr:.2f}")
        self._thread.set_gear_rider_ioa(thr)

    def _on_tracker_changed(self) -> None:
        tracker_key = self.trackerCombo.currentData()
        self._thread.set_tracker(tracker_key)
        yaml_path = tracker_yaml_path(tracker_key)
        if tracker_key == "strongsort" and not os.path.exists(yaml_path):
            self._set_status("StrongSORT config not found; falling back to ByteTrack")
        else:
            self._set_status(f"Tracker set: {tracker_key}")

    def _on_frame(self, qimg: QtGui.QImage) -> None:
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.videoLabel.setPixmap(pixmap.scaled(
            self.videoLabel.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        ))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self._stop()
        finally:
            event.accept()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1100, 800)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
