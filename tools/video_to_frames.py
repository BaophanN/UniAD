import cv2
import os
import glob

def vid_to_imgs(video_path, output_folder):
    """
    Extracts all frames from a video and saves them as images.

    :param video_path: Path to the input video file.
    :param output_folder: Folder where extracted frames will be saved.
    :fps = 25
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit when the video ends
        
        # Save every frame
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count}/{total_frames} frames and saved to {output_folder}")

def imgs_to_video(image_folder, output_video, fps=25):
    """
    Concatenates images in a folder into a video.

    :param image_folder: Path to the folder containing images.
    :param output_video: Path to save the output video file.
    :param fps: Frames per second for the output video (default: 30).
    """
    # Get list of image files in sorted order
    image_files = sorted(glob(os.path.join(image_folder, "*.jpg")) + 
                         glob(os.path.join(image_folder, "*.png")) +
                         glob(os.path.join(image_folder, "*.jpeg")))

    if not image_files:
        print("No images found in the folder.")
        return
    
    # Read the first image to get dimensions
    first_frame = cv2.imread(image_files[0])
    height, width, _ = first_frame.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write each image frame to video
    for image_file in image_files:
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Warning: Could not read {image_file}. Skipping...")
            continue
        out.write(frame)

    out.release()
    print(f"Video saved as {output_video}")


if __name__ == '__main__':
    # vid_to_imgs("input_vid/citytraffic.avi", "frames_from_vid")
    video_path = 'input_vid/citytraffic.avi'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        print('None')

    fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS
    cap.release()
    print(fps)