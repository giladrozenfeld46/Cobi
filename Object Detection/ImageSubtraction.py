import cv2
import numpy as np
import os
import time
from typing import Tuple, Optional, List

# --- Constants ---
MIN_MOTION_AREA = 400
MARKER_COLOR = (0, 0, 255)
MARKER_RADIUS = 10
MIN_CONSECUTIVE_FRAMES = 5
MAX_CENTROID_DISTANCE = 120


class MotionTracker:
    """
    Manages the state of motion tracking: buffering, coherence checks, and writing to video.
    מחלקה זו מנהלת את כל הלוגיקה של רצף התנועה, בדיקת המרחק והכתיבה לקובץ.
    """

    def __init__(self, video_writer: cv2.VideoWriter, min_frames: int, max_dist: int):
        self.writer = video_writer
        self.min_frames = min_frames
        self.max_dist = max_dist

        # State variables
        self.streak = 0
        self.buffer: List[np.ndarray] = []
        self.last_centroid: Optional[Tuple[int, int]] = None

    def update(self, frame: np.ndarray, centroid: Optional[Tuple[int, int]]):
        """
        Updates the tracker state with the new frame and centroid.
        Decides whether to buffer, write, or reset based on motion coherence.
        """
        # 1. No Motion Detected -> Reset
        if centroid is None:
            self.reset()
            return

        # 2. Check Coherence (Distance)
        if not self._is_coherent(centroid):
            print(f" -> Jump detected (Incoherent motion). Resetting streak.")
            self.reset()
            # We treat this frame as the start of a potential new streak
            self._add_to_streak(frame, centroid)
            return

        # 3. Valid Motion -> Add to Streak
        self._add_to_streak(frame, centroid)

    def _is_coherent(self, current_centroid: Tuple[int, int]) -> bool:
        """Checks if the distance between current and last centroid is valid."""
        if self.streak == 0 or self.last_centroid is None:
            return True  # First frame of a streak is always "coherent"

        dist = np.linalg.norm(np.array(current_centroid) - np.array(self.last_centroid))
        return dist <= self.max_dist

    def _add_to_streak(self, frame: np.ndarray, centroid: Tuple[int, int]):
        """Increments streak, manages buffer, and writes to file if valid."""
        self.streak += 1
        self.last_centroid = centroid

        if self.streak < self.min_frames:
            # Not enough frames yet, just buffer
            self.buffer.append(frame)

        elif self.streak == self.min_frames:
            # Threshold reached! Validate buffer and write everything
            print(f" -> Motion Confirmed! Writing buffered frames.")
            self.buffer.append(frame)
            for buf_frame in self.buffer:
                self.writer.write(buf_frame)
            self.buffer = []  # Clear buffer logic

        else:
            # Already in a valid streak, write directly
            self.writer.write(frame)

    def reset(self):
        """Resets the tracking state (clears buffer and counters)."""
        self.streak = 0
        self.buffer = []
        self.last_centroid = None


# --- Helper Functions (Stateless) ---

def initialize_video_io(input_path: str, output_path: str) -> Optional[Tuple[cv2.VideoCapture, cv2.VideoWriter]]:
    """Initializes Input/Output video streams."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video file at {input_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    try:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    except Exception as e:
        print(f"Error creating VideoWriter: {e}")
        cap.release()
        return None

    print(f"IO Initialized: {width}x{height} @ {fps}FPS")
    return cap, out


def process_frame_for_motion(current_frame: np.ndarray, prev_frame_gray: np.ndarray, threshold: int) -> Tuple[
    np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
    """
    Core Image Processing: Diff -> Threshold -> Moments -> Centroid.
    """
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(current_gray, prev_frame_gray)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    M = cv2.moments(thresh)
    centroid = None

    if M["m00"] > MIN_MOTION_AREA:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroid = (cX, cY)
        cv2.circle(current_frame, centroid, MARKER_RADIUS, MARKER_COLOR, -1)

    return current_frame, current_gray, centroid


# --- Main Function ---

def track_motion_centroid(input_path: str, output_path: str, threshold: int = 25, max_frames: Optional[int] = None):
    """
    Main Loop Driver. CLEAN & CONVENTIONAL.
    """
    if not os.path.exists(input_path):
        print("Input file not found.")
        return

    # 1. Setup IO
    io = initialize_video_io(input_path, output_path)
    if not io: return
    cap, out = io

    # 2. Setup Tracker Logic Class
    tracker = MotionTracker(out, min_frames=MIN_CONSECUTIVE_FRAMES, max_dist=MAX_CENTROID_DISTANCE)

    # 3. Read First Frame
    ret, frame = cap.read()
    if not ret: return
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    print("\nProcessing started...")
    start_time = time.time()
    frame_idx = 0

    # 4. Main Processing Loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break

        # A. Process Image (Stateless)
        processed_frame, curr_gray, centroid = process_frame_for_motion(frame, prev_gray, threshold)

        # B. Update Tracker (Stateful Logic)
        tracker.update(processed_frame, centroid)

        # C. Prepare for next iteration
        prev_gray = curr_gray.copy()
        frame_idx += 1

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Processed {frame_idx} frames in {time.time() - start_time:.2f}s")

    # 5. Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    speed_sec = end_time - start_time
    print(f"Processing complete. Validated motion segments saved to {output_path}")
    print(f"Total processing time: {speed_sec:.2f} seconds.")


# Example usage (you would need to create a dummy video or use an actual one)
# track_motion_centroid('input_video.mp4', 'output_tracked_video.avi', threshold_value=30, max_frames=30)


# --- Example Usage ---
# Ensure your 'input.mp4' file exists in the same directory or specify the full path.
INPUT_FILE = 'videos/random_staff.mp4'
OUTPUT_FILE = 'output_motion_tracking.avi'

# NOTE: You MUST have a valid video file at INPUT_FILE for this code to run.
# If you have a video, uncomment the line below:
track_motion_centroid(INPUT_FILE, OUTPUT_FILE)

"""
import cv2


def track_motion_changes(input_path, output_path, min_area=500):
    """"""
    Reads a video, detects motion by comparing frames, draws bounding boxes around
    moving objects, and saves the result.

    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the output video.
        min_area (int): Minimum area size to be considered as 'motion'.
                        Helps filter out noise.
    """"""
    # 1. Initialize video capture
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {input_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 2. Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize the first frame for comparison
    prev_frame = None

    print(f"Processing video: {input_path}...")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # 3. Pre-processing
        # Resize is optional, but converting to Grayscale is necessary for difference calculation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to smooth the image and reduce noise/false positives
        # (21, 21) is the kernel size.
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # If this is the first iteration, initialize prev_frame and continue
        if prev_frame is None:
            prev_frame = gray
            continue

        # 4. Calculate Difference
        # Compute the absolute difference between the current frame and previous frame
        frame_delta = cv2.absdiff(prev_frame, gray)

        # 5. Thresholding
        # Convert the difference to binary (black & white).
        # If difference < 25, it becomes black (0). If > 25, it becomes white (255).
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # 6. Dilate
        # Dilate the thresholded image to fill in holes within the white regions (blobs)
        # This makes the contours solid and easier to detect.
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 7. Find Contours
        # Find the outlines (contours) of the white blobs
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 8. Draw Rectangles
        for contour in contours:
            # Ignore contours that are too small (noise)
            if cv2.contourArea(contour) < min_area:
                continue

            # Compute the bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(contour)

            # Draw the rectangle on the original colored frame
            # Color: Green (0, 255, 0), Thickness: 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Optional: Add text label
            # cv2.putText(frame, "Motion Detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 9. Write the frame (with drawings) to the output video
        out.write(frame)

        # Update the previous frame for the next iteration
        prev_frame = gray

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done. Video saved to {output_path}")


# --- Execution Example ---
input_video = 'papertrimmed.mp4'
output_video = 'output_tracked.mp4'

track_motion_changes(input_video, output_video, min_area=500)
    """