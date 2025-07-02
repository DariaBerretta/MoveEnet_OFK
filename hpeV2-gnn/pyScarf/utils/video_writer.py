import cv2
import os

def create_video_writer(output_path, resolution, fps=100, codec='mp4v'):
    """
    Initialize an OpenCV VideoWriter.

    Args:
        output_path (str): Full path to output video (e.g., 'scarf_output.avi').
        resolution (tuple): (width, height) of the video.
        fps (int): Frames per second.
        codec (str): FourCC codec string (default 'mp4v').

    Returns:
        cv2.VideoWriter: Ready to use.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    return writer


def write_frame(writer, img_gray):
    """
    Write a grayscale frame to the video. Handles BGR conversion.

    Args:
        writer (cv2.VideoWriter): OpenCV video writer.
        img_gray (np.ndarray): Grayscale image (uint8).
    """
    if len(img_gray.shape) == 2:  # grayscale to BGR
        img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img_gray  # already BGR
    writer.write(img_color)
