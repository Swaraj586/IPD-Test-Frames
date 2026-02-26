import cv2
import os

def extract_frames(video_path, output_folder):
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: The video file '{video_path}' was not found.")
        return

    # Create the output folder if it doesn't already exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0

    print("Starting frame extraction...")

    while True:
        # Read the next frame
        success, frame = video_capture.read()

        # If success is False, we have reached the end of the video
        if not success:
            break

        # Construct the output file path (e.g., frame_0000.jpg, frame_0001.jpg)
        # Using zfill to pad the numbers with zeros for better file sorting
        frame_filename = f"frame_{str(frame_count).zfill(4)}.jpg"
        output_path = os.path.join(output_folder, frame_filename)

        # Save the frame as an image file
        cv2.imwrite(output_path, frame)

        frame_count += 1

    # Release the video capture object
    video_capture.release()
    print(f"Extraction complete! Saved {frame_count} frames to '{output_folder}'.")

# --- Example Usage ---
if __name__ == "__main__":
    # Replace these paths with your actual paths
    input_video1 = "videos/obj_bench.mp4"
    input_video2 = "videos/obj_class.mp4"
    input_video3 = "videos/obj_passage.mp4"

    output_directory = "OutputFrames"

    extract_frames(input_video1, output_directory)
    extract_frames(input_video2, output_directory)
    extract_frames(input_video3, output_directory)