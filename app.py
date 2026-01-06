import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from ultralytics import YOLO


@dataclass
class AppConfig:
    conf: float = 0.25
    iou: float = 0.45
    tracker_key: str = "botsort"


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
        self._model_path: Optional[str] = None
        self._config = AppConfig()
        self._stop = False
        self._paused = True
        self._restart_requested = False
        self._model: Optional[YOLO] = None

        self._frame_delay_ms: int = 1

    def set_video_path(self, path: Optional[str]) -> None:
        self._video_path = path

    def set_model_path(self, path: Optional[str]) -> None:
        self._model_path = path

    def set_conf(self, conf: float) -> None:
        self._config.conf = clamp01(conf)

    def set_iou(self, iou: float) -> None:
        self._config.iou = clamp01(iou)

    def set_tracker(self, tracker_key: str) -> None:
        self._config.tracker_key = tracker_key
        self._restart_requested = True

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
        if not self._model_path:
            raise RuntimeError("No model selected")
        self._emit_status(f"Loading model: {os.path.basename(self._model_path)}")
        self._model = YOLO(self._model_path)

    def _annotate(self, frame_bgr: np.ndarray, result) -> np.ndarray:
        annotated = frame_bgr.copy()

        names = getattr(result, "names", None) or {}

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return annotated

        xyxy = getattr(boxes, "xyxy", None)
        cls = getattr(boxes, "cls", None)
        conf = getattr(boxes, "conf", None)
        ids = getattr(boxes, "id", None)

        if xyxy is None or cls is None:
            return annotated

        xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
        cls = cls.cpu().numpy() if hasattr(cls, "cpu") else np.asarray(cls)

        conf_arr = None
        if conf is not None:
            conf_arr = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)

        id_arr = None
        if ids is not None:
            try:
                id_arr = ids.cpu().numpy() if hasattr(ids, "cpu") else np.asarray(ids)
            except Exception:
                id_arr = None

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
            class_id = int(cls[i])

            color = CLASS_COLORS_BGR.get(class_id, (0, 255, 255))

            label_name = names.get(class_id, CLASS_NAMES_DEFAULT.get(class_id, str(class_id)))
            label = label_name

            if conf_arr is not None:
                label += f" {conf_arr[i]:.2f}"

            if class_id == 1 and id_arr is not None and i < len(id_arr) and id_arr[i] is not None:
                try:
                    track_id = int(id_arr[i])
                    label += f" ID:{track_id}"
                except Exception:
                    pass

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
                self._emit_status("Restarted video due to tracker change")

            ok, frame = cap.read()
            if not ok:
                self._emit_status("End of video")
                self._paused = True
                continue

            tracker_path = tracker_yaml_path(self._config.tracker_key)
            if self._config.tracker_key == "strongsort" and not os.path.exists(tracker_path):
                tracker_path = tracker_yaml_path("bytetrack")

            try:
                results = self._model.track(
                    frame,
                    conf=self._config.conf,
                    iou=self._config.iou,
                    persist=True,
                    tracker=tracker_path,
                    verbose=False,
                )

                result0 = results[0] if isinstance(results, (list, tuple)) else results
                annotated = self._annotate(frame, result0)
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
        self._model_path: Optional[str] = None

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
        self.btnLoadModel = QtWidgets.QPushButton("Load Model (.pt)")

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

        self.confValue = QtWidgets.QLabel("0.25")
        self.iouValue = QtWidgets.QLabel("0.45")

        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.btnLoadVideo, 0, 0)
        grid.addWidget(self.btnLoadModel, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Tracker"), 0, 2)
        grid.addWidget(self.trackerCombo, 0, 3)

        grid.addWidget(QtWidgets.QLabel("Conf Threshold"), 1, 0)
        grid.addWidget(self.confSlider, 1, 1, 1, 2)
        grid.addWidget(self.confValue, 1, 3)

        grid.addWidget(QtWidgets.QLabel("IoU Threshold"), 2, 0)
        grid.addWidget(self.iouSlider, 2, 1, 1, 2)
        grid.addWidget(self.iouValue, 2, 3)

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
        self.btnLoadModel.clicked.connect(self._pick_model)

        self.btnPlay.clicked.connect(self._play)
        self.btnPause.clicked.connect(self._pause)
        self.btnStop.clicked.connect(self._stop)

        self.confSlider.valueChanged.connect(self._on_conf_changed)
        self.iouSlider.valueChanged.connect(self._on_iou_changed)
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

    def _pick_model(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Model", "", "PyTorch Model (*.pt);;All Files (*)"
        )
        if not path:
            return
        self._model_path = path
        self._thread.set_model_path(path)
        self._set_status(f"Model selected: {os.path.basename(path)}")

    def _ensure_thread_started(self) -> None:
        if self._thread.isRunning():
            return
        self._thread.start()

    def _play(self) -> None:
        if not self._video_path or not self._model_path:
            self._set_status("Select a video and a model first")
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
