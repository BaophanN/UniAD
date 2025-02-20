import cv2
import os
import glob
from nuscenes.nuscenes import NuScenes

import cv2
import os

import cv2
import os
import numpy as np

def vid_to_81_imgs(video_path, output_folder):
    """
    Extract exactly 81 frames evenly spaced from a video.

    :param video_path: Path to the input video file.
    :param output_folder: Folder where extracted frames will be saved.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    frame_indices = np.linspace(0, total_frames - 1, 81, dtype=int)  # Select 81 evenly spaced frames

    frame_count = 0
    saved_frames = 0

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break  # Exit when video ends
        
        if frame_count in frame_indices:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frames:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1
        
        frame_count += 1

    cap.release()
    print(f"Extracted {saved_frames} frames from {total_frames} and saved to {output_folder}")

# Example usage


def vid_to_imgs(video_path, output_folder, target_fps=12):
    """
    Extracts frames from a video at a specified FPS and saves them as images.

    :param video_path: Path to the input video file.
    :param output_folder: Folder where extracted frames will be saved.
    :param target_fps: Number of frames per second to save (default: 12).
    :citytraffic.avi: 24s: 299 frames 
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the original FPS of the video
    frame_interval = max(1, video_fps // target_fps)  # Determine frame skipping interval

    frame_count = 0
    saved_frames = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit when the video ends
        
        # Save only every `frame_interval` frame
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frames:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1
        frame_count += 1

    cap.release()
    print(f"Extracted {saved_frames} frames from {total_frames} frames and saved to {output_folder}")

# Example usage
# vid_to_imgs("example.mp4", "output_frames", target_fps=12)


def imgs_to_video(image_folder, output_video, fps=4):
    """
    Concatenates images in a folder into a video.

    :param image_folder: Path to the folder containing images.
    :param output_video: Path to save the output video file.
    :param fps: Frames per second for the output video (default: 30).
    """
    # Get list of image files in sorted order
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")) + 
                         glob.glob(os.path.join(image_folder, "*.png")) +
                         glob.glob(os.path.join(image_folder, "*.jpeg")))

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
    i = 0 
    for image_file in image_files:
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Warning: Could not read {image_file}. Skipping...")
            continue
        print(f'Processing frame {i}')
        out.write(frame)
        i += 1

    out.release()
    print(f"Video saved as {output_video}")

def print_scenes():
    # Load the NuScenes mini dataset
    nusc = NuScenes(version="v1.0-mini", dataroot="data/nuscenes", verbose=True)

    # Target scenes: val scenes
    target_scenes = {"scene-0103", "scene-0916"}

    # Get the tokens for the target scenes
    scene_tokens = [scene["token"] for scene in nusc.scene if scene["name"] in target_scenes]

    # Find the first sample for each scene
    for scene in nusc.scene:
        if scene["token"] in scene_tokens:
            first_sample_token = scene["first_sample_token"]
            
            # Traverse through all samples in the scene
            while first_sample_token:
                sample = nusc.get("sample", first_sample_token)

                # Get the front camera data
                cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
                cam_front_filepath = cam_front_data["filename"]
                cam_front_filename = os.path.basename(cam_front_filepath)
                # print(f"Scene {scene['name']} Front View Image: {cam_front_filepath}")
                print(f"{cam_front_filename}")

                # Move to next sample in the scene
                first_sample_token = sample["next"]

import os
import glob
from nuscenes.nuscenes import NuScenes
import cv2
import os
import glob

def resize_images(input_folder, output_size=(1600, 900)):
    """
    Resize all images in a folder to the specified size.

    :param input_folder: Path to the folder containing images.
    :param output_size: Tuple (width, height), default is (1600, 900).
    """
    # Get all image files (JPG, PNG, JPEG)
    image_files = glob.glob(os.path.join(input_folder, "*.jpg")) + \
                  glob.glob(os.path.join(input_folder, "*.png")) + \
                  glob.glob(os.path.join(input_folder, "*.jpeg"))

    if not image_files:
        print("No images found in the folder.")
        return

    for img_path in image_files:
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Could not read {img_path}")
            continue

        # Resize image
        resized_image = cv2.resize(image, output_size)

        # Overwrite the original file
        cv2.imwrite(img_path, resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print(f"Resized {img_path} to {output_size}")

    print("Resizing complete!")

# Example Usage:

def rename_extracted_frames(output_folder="output_frames"):
    """
    Rename 81 extracted frames (frame_0000.jpg, frame_0001.jpg, ...) to NuScenes CAM_FRONT filenames.
    
    :param output_folder: Folder where extracted frames are stored.
    """
    # Load NuScenes mini dataset
    nusc = NuScenes(version="v1.0-mini", dataroot="data/nuscenes", verbose=True)

    # Target scenes (validation set)
    target_scenes = {"scene-0103", "scene-0916"}

    # Get scene tokens
    scene_tokens = {scene["token"]: scene["name"] for scene in nusc.scene if scene["name"] in target_scenes}

    # Collect filenames for `CAM_FRONT`
    nuscenes_filenames = []

    # Traverse all samples in target scenes
    for scene in nusc.scene:
        if scene["token"] in scene_tokens:
            first_sample_token = scene["first_sample_token"]

            while first_sample_token:
                sample = nusc.get("sample", first_sample_token)
                cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
                cam_front_filename = os.path.basename(cam_front_data["filename"])  # Extract filename
                nuscenes_filenames.append(cam_front_filename)
                
                # Move to next sample in the scene
                first_sample_token = sample["next"]

    # Ensure exactly 81 filenames are collected
    if len(nuscenes_filenames) != 81:
        print(f"Error: Expected 81 NuScenes filenames but found {len(nuscenes_filenames)}")
        return

    # Get sorted list of extracted frames (frame_0000.jpg, frame_0001.jpg, ...)
    extracted_frames = sorted(glob.glob(os.path.join(output_folder, "frame_*.jpg")))

    # Ensure we have exactly 81 extracted frames
    if len(extracted_frames) != 81:
        print(f"Error: Expected 81 extracted frames but found {len(extracted_frames)}")
        return

    # Rename extracted frames to match NuScenes filenames
    for extracted_frame, new_filename in zip(extracted_frames, nuscenes_filenames):
        new_path = os.path.join(output_folder, new_filename)  # New name with only filename
        os.rename(extracted_frame, new_path)
        print(f"Renamed {extracted_frame} -> {new_path}")

    print("Renaming complete!")

# Call the function


if __name__ == '__main__':
    # vid_to_imgs("input_vid/citytraffic.avi", "frames_from_vid")
    # rename_extracted_frames("81_frames_from_vid")
    resize_images("CAM_FRONT")  # Change "output_frames" to your folder path

    # imgs_to_video("data/nuscenes/sweeps/CAM_BACK","concat_CAM_BACK.mp4")
    # print_scenes()  
    # video_path = 'input_vid/citytraffic.avi'
    # cap = cv2.VideoCapture(video_path)
    
    # if not cap.isOpened():
    #     print("Error: Could not open video.")
    #     print('None')

    # fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS
    # cap.release()
    # print(fps)
    # vid_to_81_imgs("input_vid/citytraffic.avi", "81_frames_from_vid")