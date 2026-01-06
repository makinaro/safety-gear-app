import os
import sys

def list_video_files(directory):
    """List all video files in the specified directory."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return []
    
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory.")
        return []
    
    video_files = []
    try:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(file)
                if ext.lower() in video_extensions:
                    video_files.append(file_path)
    except Exception as e:
        print(f"Error reading directory: {e}")
        return []
    
    return video_files

if __name__ == "__main__":
    # Get directory from command line argument
    if len(sys.argv) < 2:
        print("Usage: python list_videos.py <directory_path>")
        print("\nExample:")
        print('  python list_videos.py "C:\\path\\to\\videos"')
        sys.exit(1)
    
    directory = sys.argv[1]
    
    print(f"Searching for video files in:\n{directory}\n")
    
    video_files = list_video_files(directory)
    
    if not video_files:
        print("No video files found!")
    else:
        print(f"Found {len(video_files)} video file(s):\n")
        for i, video in enumerate(video_files, 1):
            print(f"{i}. {os.path.basename(video)}")
            print(f"   Full path: {video}\n")
        
        print("\nTo process a video in the GUI:")
        print("1. Run: python app_gui.py")
        print("2. Select 'Video File' option")
        print('3. Click "Browse..." and select your video')
        print("\nOr use command line:")
        print(f'python app.py --source "{video_files[0]}"')
