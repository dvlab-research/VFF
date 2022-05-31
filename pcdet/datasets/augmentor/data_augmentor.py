from functools import partial

import numpy as np
import cv2

from ...utils import common_utils
from . import augmentor_utils, database_sampler


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST
        self.image_aug = augmentor_configs.get("IMAGE_AUG", []) # use joint image augmentation
        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points, matrix = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )
            if 'flip' not in self.image_aug:
                data_dict['aug_matrix'] = np.matmul(matrix, data_dict['aug_matrix'])
            else:
                data_dict['aug_points'] = points.copy()
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, matrix = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )
        if 'rotate' not in self.image_aug:
            data_dict['aug_matrix'] = np.matmul(matrix, data_dict['aug_matrix'])
        else:
            data_dict['aug_points'] = points.copy()
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points, matrix = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
        )

        if 'rescale' not in self.image_aug:
            data_dict['aug_matrix'] = np.matmul(matrix, data_dict['aug_matrix'])
        else:
            data_dict['aug_points'] = points.copy()
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']
            images, depth_maps, gt_boxes = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                images, depth_maps, gt_boxes, calib,
            )

        data_dict['images'] = images
        data_dict['depth_maps'] = depth_maps
        data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        # Generate augmentation matrix
        if 'aug_matrix' not in data_dict.keys():
            data_dict['aug_matrix'] = np.eye(3)
        
        # Sample Anchor Points before GT-Sample
        random_idx = np.random.permutation(np.arange(0,len(data_dict['points'])))[:100]
        points_raw = data_dict['points'][random_idx,:3].copy()
        points2d_raw, depth_raw = data_dict['calib'].lidar_to_img(points_raw)
        points2d_raw = points2d_raw.reshape(-1, 1, 2).astype(np.int)

        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)
            if len(self.image_aug)>0 and isinstance(cur_augmentor, database_sampler.DataBaseSampler):
                random_idx = np.random.permutation(np.arange(0,len(data_dict['points'])))[:100]
                points_raw = data_dict['points'][random_idx,:3].copy()
                points2d_raw, depth_raw = data_dict['calib'].lidar_to_img(points_raw)
                points2d_raw = points2d_raw.reshape(-1, 1, 2).astype(np.int)

        # Estimate the affine matrix according to the point transformation
        if 'aug_points' in data_dict.keys():
            points_aug = data_dict['aug_points'][random_idx,:3].copy()
            points2d_aug, depth_aug = data_dict['calib'].lidar_to_img(points_aug)
            points2d_aug = points2d_aug.reshape(-1, 1, 2).astype(np.int)
            ransacReprojThreshold = 10
            images = data_dict["images"]
            H, status = cv2.findHomography(points2d_aug, points2d_raw, cv2.RANSAC, ransacReprojThreshold)
            data_dict["images"] = cv2.warpPerspective(images, H, (images.shape[1], images.shape[0]),
                                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            data_dict["affine_matrix"] = H
            data_dict.pop('aug_points')

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]

            data_dict.pop('gt_boxes_mask')
        return data_dict
