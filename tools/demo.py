import argparse
import glob
import pickle
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
from skimage import io
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.datasets.kitti import kitti_utils
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, calibration_kitti
from visual_utils import visualize_utils as V

# from pyvirtualdisplay import Display
# display = Display(visible=False, size=(1280, 1024))
# display.start()
# mlab.options.offscreen = True

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin', 
                 depth=False, image=False, calib=False, info_path=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list
        self.depth = depth
        self.image = image
        self.calib = calib
        if info_path is not None:
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
            self.shape_dict = {_info['image']['image_idx']:_info['image']['image_shape'] for _info in infos}
    
    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        if self.depth:
            depth_file = str(self.sample_file_list[index]).replace('velodyne','depth_2')
            depth_file = depth_file.replace(self.ext, '.png')
            depth = io.imread(depth_file)
            depth = depth.astype(np.float32)
            depth /= 256.0
            input_dict['depth_maps'] = depth

        if self.image:
            image_file = str(self.sample_file_list[index]).replace('velodyne','image_2')
            image_file = image_file.replace(self.ext, '.png')
            img_idx = str(self.sample_file_list[index]).split('/')[-1][:-len(self.ext)]
            image = io.imread(image_file)
            image = image.astype(np.float32)
            image /= 255.0
            input_dict['images'] = image
            input_dict['image_shape'] = self.shape_dict[img_idx]

        if self.calib:
            calib_file = str(self.sample_file_list[index]).replace('velodyne','calib')
            calib_file = calib_file.replace(self.ext, '.txt')
            calibration = calibration_kitti.Calibration(calib_file)
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calibration)

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--save_file', type=str, default=None, help='specify name for save figure')
    parser.add_argument('--info', type=str, default=None, help='info path for KITTI')
    parser.add_argument('--depth', type=bool, default=False, help='using depth for Demo')
    parser.add_argument('--image', type=bool, default=False, help='using image for Demo')
    parser.add_argument('--calib', type=bool, default=False, help='using calib for Demo')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger, info_path=args.info,
        depth=args.depth, image=args.image, calib=args.calib
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                ptbg_color=(1,1,1), ptfg_color=(1, 0.667, 0.118)
            )
            if args.save_file is not None:
                mlab.savefig(args.save_file)
            else:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
