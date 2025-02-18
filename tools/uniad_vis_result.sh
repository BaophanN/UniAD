#!/bin/bash
# ./tools/uniad_dist_eval.sh ./projects/configs/stage1_track_map/base_track_map.py ./ckpts/uniad_base_track_map.pth 1
# ./tools/uniad_dist_eval.sh ./projects/configs/stage2_e2e/base_e2e.py ./ckpts/uniad_base_e2e.pth 1
#
PATH_TO_YOUR_PREDISION_RESULT_PKL=output/results.pkl
PATH_TO_YOUR_OUTPUT_FOLDER=vids/run_cam_front_refine/image
FILENAME_OF_OUTPUT_VIDEO=base_e2e_test_camfront_demo_refine_again.avi
python ./tools/analysis_tools/visualize/run_cam_front.py \
    --predroot $PATH_TO_YOUR_PREDISION_RESULT_PKL \
    --out_folder $PATH_TO_YOUR_OUTPUT_FOLDER \
    --demo_video $FILENAME_OF_OUTPUT_VIDEO \
    --project_to_cam True