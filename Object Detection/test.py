import cv2
import time
import os

# --- Configuration ---
# Camera Index: 0 is typically the first USB camera on Linux/Raspberry Pi.
# If you have multiple cameras, try 1, 2, etc.
CAMERA_INDEX = 1
OUTPUT_FILENAME = "output_video.avi"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30.0
# The fourcc code for video compression. 'XVID' is widely compatible.
FOURCC_CODE = cv2.VideoWriter_fourcc(*'XVID')


# --- Main Application ---
def record_video():
    """Initializes the camera, captures video, and saves it to a file."""

    # 1. Initialize Video Capture Object
    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print(f"Error: Cannot open camera with index {CAMERA_INDEX}.")
        print("1. Check if the camera is connected.")
        print("2. Verify the correct CAMERA_INDEX (try 0, 1, or 2).")
        print("3. Ensure you ran 'sudo modprobe uvcvideo' if driver was missing.")
        return

    # Set preferred resolution and FPS (Note: the camera may override these)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Get the actual, final resolution used by the camera
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera successfully opened.")
    print(f"Resolution: {actual_width}x{actual_height}, FPS: {actual_fps:.2f}")

    # 2. Initialize Video Writer Object
    # Creates an object to save the captured video stream.
    out = cv2.VideoWriter(OUTPUT_FILENAME, FOURCC_CODE, actual_fps, (actual_width, actual_height))

    # Check if the writer initialized correctly
    if not out.isOpened():
        print(f"Error: Could not open video writer for file {OUTPUT_FILENAME}")
        print("Please check if the file path is writeable.")
        # Release the camera before exiting
        cap.release()
        return

    # Start loop flag
    recording = False
    print("\n--- Controls ---")
    print("Press 'S' to START recording.")
    print("Press 'Q' to STOP recording and EXIT.")
    print("----------------")

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # If frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Add overlay text
            status_text = "STATUS: READY (Press 'S' to Start)"
            if recording:
                status_text = "STATUS: RECORDING..."
                # Write the current frame to the output file
                out.write(frame)

            # Display the status on the frame
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('Live Camera Feed (Press Q to Exit)', frame)

            # --- Key Press Handling ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Quit and exit loop
                print("\nQuit key ('Q') pressed. Stopping and exiting...")
                break

            if key == ord('s'):
                if not recording:
                    # Start recording
                    start_time = time.time()
                    print(f"Recording started. Saving to {OUTPUT_FILENAME}")
                    recording = True

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # 3. When everything is done, release the capture and writer objects
        cap.release()
        out.release()

        # Close all the OpenCV windows
        cv2.destroyAllWindows()

        if os.path.exists(OUTPUT_FILENAME):
            print(f"\nVideo saved successfully as: {os.path.abspath(OUTPUT_FILENAME)}")
        else:
            print("\nRecording stopped, but file was not created. Check for errors.")


if __name__ == '__main__':
    # Check if the required library is installed
    try:
        import cv2
    except ImportError:
        print("Error: The 'opencv-python' library is not installed.")
        print("Please install it using: pip install opencv-python")
    else:
        record_video()