import numpy as np 
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from scipy.spatial.transform import Rotation
def load_colmap_poses(file_path):
    """
    Load pose data from a COLMAP output file.
    The file is expected to have a header with comment lines,
    followed by two lines per image:
      - The first line of each pair contains pose data:
        IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
      - The second line contains 2D keypoints (which will be skipped)
    
    :param file_path: Path to the COLMAP output .txt file.
    :return: List of dictionaries with pose data.
    """
    poses = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Filter out comment and empty lines
    data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    
    # Since the data is arranged in pairs (pose line, then keypoints line),
    # iterate with a step of 2 to process only the pose lines.
    for i in range(0, len(data_lines), 2):

        pose_line = data_lines[i]
        tokens = pose_line.split()
        # Expecting at least 10 tokens: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        if len(tokens) < 10:
            continue
        
        image_id = tokens[0]
        qw, qx, qy, qz = map(float, tokens[1:5])
        tx, ty, tz = map(float, tokens[5:8])
        camera_id = tokens[8]
        name = tokens[9]
        
        pose_data = {
            'image_id': image_id,
            'rotation': (qw, qx, qy, qz),
            'translation': (tx, ty, tz),
            'camera_id': camera_id,
            'name': name
        }
        poses.append(pose_data)
        if i == 10: 
            break
        
    return poses

def colmap_to_nuscenes(colmap_rotation, colmap_translation):
    qw, qx, qy, qz = colmap_rotation 
    R = Quaternion(qw,qx,qy,qz).rotation_matrix 

    # Compute actual ego position (translation) in world coordinates 
    T = np.array(colmap_translation)  # (TX, TY, TZ)
    ego2global_translation = (-R.T @ T).tolist() # Correct position in world space 

    # Compute ego to global rotation (invert COLMAP's rotation) 
    R_ego2global = R.T 
    ego2global_rotation = Quaternion(matrix=R_ego2global)
    return ego2global_translation, ego2global_rotation


def update_can_bus(input_dict, colmap_poses):
    img_name = input_dict["img_file_name"]
    if img_name in colmap_poses: 
        rotation, translation = colmap_poses[img_name] 
        input_dict['can_bus'][:3] = translation 
        input_dict['can_bus'][3:7] = rotation.elements

        yaw_angle = np.arctan2(2.0 * (rotation.w * rotation.z + rotation.x * rotation.y), 1.0 - 2.0 * (rotation.y**2 + rotation.z**2)) * 180 / np.pi

        input_dict['can_bus'][-2] = yaw_angle / 180 * np.pi  # Keep radian format
        input_dict['can_bus'][-1] = yaw_angle  # Store angle for shift computation


# poses = load_colmap_poses("/workspace/datasets/colmap/sparse/0/images.txt")
# for pose in poses:
#     print(pose)
#     print('\n')
# Example COLMAP pose
if __name__ == "__main__":
    colmap_quaternion = (0.966281, -0.121643, 0.226905, 0.004220)  # (QW, QX, QY, QZ)
    colmap_translation = (-2.272727, -0.459435, 5.525815)  # (TX, TY, TZ)

    # Convert to NuScenes format
    ego2global_translation, ego2global_rotation = colmap_to_nuscenes(colmap_quaternion, colmap_translation)

    print("Ego2Global Translation:", ego2global_translation, type(ego2global_translation))
    print("Ego2Global Rotation:", ego2global_rotation, type(ego2global_rotation))