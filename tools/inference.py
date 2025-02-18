import sys
sys.path.insert(0,'/workspace/source/UniAD')
import mmcv
import torch
import numpy as np
from mmcv import Config
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint
import os 
def inference():

    # Load model config
    config_path = "projects/configs/stage2_e2e/base_e2e.py"  # Set your config file path
    checkpoint_path = "ckpts/uniad_base_e2e.pth"  # Set your model checkpoint path
    cfg = Config.fromfile(config_path)

    # Modify config for single image inference
    cfg.model.video_test_mode = False  # Disable video test mode since we use a single image

    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # Build model
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    model.eval()
    model.cuda()

    # Load checkpoint
    load_checkpoint(model, checkpoint_path, map_location="cpu")

    # Load image
    image_path = "frames_from_vid/frame_0001.jpg"  # Set your image path
    img = mmcv.imread(image_path)
    img = mmcv.imresize(img, (1600,900))

    # Convert to tensor and move to GPU
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda()

    # Construct dummy `img_metas`
    img_metas = [[{
        'img_shape': img.shape,
        'ori_shape': img.shape,
        'pad_shape': img.shape,
        'scale_factor': 1.0,
        'flip': False,
        'scene_token': 'dummy_scene_token',
        'sample_idx': 'dummy_sample_idx',
        'can_bus': np.zeros(6).tolist(),  # Dummy ego-motion data
    }]]

    # Run inference
    with torch.no_grad():
        results = model.forward_test(img=[img_tensor], img_metas=img_metas)

    # Print results
    print("Inference Results:", results)

    # Example: If the result contains bounding boxes
    if 'pts_bbox' in results[0]:
        bboxes = results[0]['pts_bbox']['boxes_3d']
        scores = results[0]['pts_bbox']['scores_3d']
        print(f"Detected {len(bboxes)} objects with scores:", scores)



if __name__ == '__main__':
    inference()
    # Example usage
