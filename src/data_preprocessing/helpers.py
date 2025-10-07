import cv2 

def get_video_info(video_path):
    """
    Get video metadata including duration, frame count, fps, and dimensions.
    Returns a dictionary with this information.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate duration in seconds
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "frame_count": frame_count,
        "fps": fps,
        "duration_seconds": duration,
        "width": width,
        "height": height
    }