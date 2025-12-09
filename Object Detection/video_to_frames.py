import cv2
import os
import sys

# --- Configuration ---

# REQUIRED: Path to your input video file.
# You can change this to a file path like 'C:/Users/YourName/Videos/input.mp4'
VIDEO_PATH = 'output_motion_tracking.avi'

# REQUIRED: Name of the folder where the extracted images will be saved.
OUTPUT_DIR = 'output_frames'

# --- Frame Extraction Logic ---

def extract_frames(video_path, output_dir):
    """
    Reads a video file and saves every frame as a JPEG image in the specified output directory.
    """
    print(f"Attempting to open video: {video_path}")

    # 1. Initialize video capture object
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at path: {video_path}")
        print("Please check if the file exists and the path is correct.")
        sys.exit()

    # 2. Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Using existing output directory: {output_dir}")

    frame_count = 0

    print("Starting frame extraction...")

    # 3. Loop through all frames
    while True:
        # Read the next frame
        ret, frame = cap.read()

        # If 'ret' is False, we have reached the end of the video
        if not ret:
            break

        # Generate the output filename (e.g., 'output_frames/frame_00001.jpg')
        frame_filename = os.path.join(output_dir, f'frame_{frame_count:05d}.jpg')

        # Save the frame as a JPEG image
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

        # Simple progress indicator
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")


    # 4. Release the video capture object and clean up
    cap.release()
    print(f"\nExtraction complete! Total frames saved: {frame_count}")
    print(f"Files are located in the '{output_dir}' directory.")

# Execute the function
if __name__ == '__main__':
    # NOTE: You must place a video file named 'input_video.mp4' in the same directory
    # as this script, or update the VIDEO_PATH variable above.
    extract_frames(VIDEO_PATH, OUTPUT_DIR)