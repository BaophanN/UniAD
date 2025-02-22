import pickle

# -------------------------------
# CONFIGURATION
# -------------------------------
nuscenes_metadata_path = "data/infos-test/nuscenes_infos_temporal_test.pkl"  # Path to NuScenes metadata
nuscenes_metadata_path = "data/infos/nuscenes_infos_temporal_val.pkl"  # Path to NuScenes metadata

output_txt_path = "nuscenes_val_metadata.txt"  # Output text file

# -------------------------------
# LOAD NUSCENES METADATA
# -------------------------------
with open(nuscenes_metadata_path, "rb") as f:
    nuscenes_metadata = pickle.load(f)

# Extract metadata entries
# for var, name in [(nuscenes_metadata['infos'], 'img_metas')]:
#     if not isinstance(var, list):
#         raise TypeError('{} must be a list, but got {}'.format(
#             name, type(var)))
#     print('var: ',var,'name: ',name)
# exit()
metadata_entries = nuscenes_metadata["infos"]
# print(metadata_entries);exit()
# -------------------------------
# WRITE TO TXT FILE
# -------------------------------
with open(output_txt_path, "w") as f:
    for idx, meta in enumerate(metadata_entries):
        # print(meta.keys());exit()
        f.write(f"Frame {idx}:\n")
        f.write(f"  Token: {meta['token']}\n")
        f.write(f"  Scene Token: {meta['scene_token']}\n")
        f.write(f"  Sweeps: {meta['sweeps']}\n")
        f.write(f"  cams: {meta['cams']}\n")
        f.write(f"  Timestamp: {meta['timestamp']}\n")
        f.write(f"  Frame Index: {meta['frame_idx']}\n")
        f.write(f"  Previous Sample: {meta['prev']}\n")
        f.write(f"  Next Sample: {meta['next']}\n")
        f.write(f"  LiDAR Path: {meta['lidar_path']}\n")
        # f.write(f"  LiDAR to Ego Translation: {meta['lidar2ego_translation']}\n")
        # f.write(f"  LiDAR to Ego Rotation: {meta['lidar2ego_rotation']}\n")
        f.write(f"  Ego to Global Translation: {meta['ego2global_translation']}\n")
        f.write(f"  Ego to Global Rotation: {meta['ego2global_rotation']}\n")
        # f.write(f"  Cam Intrinsic: {meta['cam_intrinsic']}\n")
        f.write(f"  CAN Bus Data: {meta['can_bus']}\n")

        # Ground Truth Information
        # f.write(f"### GROUND TRUTH ###\n")
        # f.write(f"  Ground Truth Boxes: {meta['gt_boxes']}\n")
        # f.write(f"  Ground Truth Names: {meta['gt_names']}\n")
        # f.write(f"  Ground Truth Velocity: {meta['gt_velocity']}\n")
        # f.write(f"  Ground Truth Indices: {meta['gt_inds']}\n")
        # f.write(f"  Ground Truth Instance Tokens: {meta['gt_ins_tokens']}\n")
        
        # Additional Information
        # f.write(f'### ADDITIONAL INFO ###\n')
        # f.write(f"  Number of LiDAR Points: {meta['num_lidar_pts']}\n")
        # f.write(f"  Number of Radar Points: {meta['num_radar_pts']}\n")
        # f.write(f"  Valid Flag: {meta['valid_flag']}\n")
        
        # Future Trajectory Data
        # f.write(f'### FUTURE TRAJECTORY ###\n')
        # f.write(f"  Future Trajectory: {meta['fut_traj']}\n")
        # f.write(f"  Future Trajectory Valid Mask: {meta['fut_traj_valid_mask']}\n")
        # f.write(f"  Visibility Tokens: {meta['visibility_tokens']}\n")
        
        f.write("-" * 80 + "\n")  # Separator for readability

print(f"Metadata successfully written to {output_txt_path}")
