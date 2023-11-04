import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, Dict, List, Union

import cv2
import numpy as np
import omegaconf
import open3d as o3d
from omegaconf import OmegaConf
from loguru import logger

from common.experiment_base import ExperimentRealTransforms, config_completion, convert_dict
from third_party import PhoXiCamera


class FileCamera:
    _pcd_path_fn: Dict[str, Callable[[str], str]]
    _rgb_path_fn: Dict[str, Callable[[str], str]]
    _stages: Optional[List[str]] = None
    _started: bool = False

    pcd_frame_idx: int = 0
    rgb_frame_idx: int = 0
    pcd_sub_frame_idx: int = 0
    rgb_sub_frame_idx: int = 0

    target_dir: str = ''
    log_list: List[str]

    external_intrinsics: np.ndarray
    external_extrinsics: np.ndarray
    external_distortion_coeff: np.ndarray
    external_camera_resolution: np.ndarray
    transforms: ExperimentRealTransforms

    def __init__(self,
                 target_dir: str,
                 external_calibration_path: str,
                 transforms: ExperimentRealTransforms,
                 stages: Optional[List[str]] = None,
                 pcd_path_fn: Optional[Dict[str, Callable[[str], str]]] = None,
                 rgb_path_fn: Optional[Dict[str, Callable[[str], str]]] = None,
                 **kwargs):
        self.target_dir = target_dir

        external_intrinsics, external_extrinsics, external_distortion_coeff, external_camera_resolution = \
            PhoXiCamera.load_calibration_txt(external_calibration_path)
        self.external_intrinsics = external_intrinsics
        self.external_extrinsics = external_extrinsics
        self.external_distortion_coeff = external_distortion_coeff
        self.external_camera_resolution = external_camera_resolution
        self.transforms = transforms

        self._pcd_path_fn = {
            'begin': lambda x: osp.join(self.target_dir, x, 'pcd', 'raw_begin.npz'),
            'end': lambda x: osp.join(self.target_dir, x, 'pcd', 'raw_end.npz'),
        } if pcd_path_fn is None else pcd_path_fn
        self._rgb_path_fn: Dict[str, Callable[[str], str]] = {
            'begin': lambda x: osp.join(self.target_dir, x, 'rgb', 'begin.jpg'),
            'end': lambda x: osp.join(self.target_dir, x, 'rgb', 'end.jpg'),
        } if rgb_path_fn is None else rgb_path_fn

        self._stages: List[str] = ['begin', 'end'] if stages is None else stages
        assert len(self._pcd_path_fn) == len(self._rgb_path_fn) == len(self._stages)

        self._reset()

    def start(self) -> Optional[Exception]:
        """scan log dir and return all log files
        Returns:

            - exception if any
        """
        self._reset()

        if self.target_dir == "" or self.target_dir is None or not osp.exists(self.target_dir):
            self.log_list = []
            return Exception(f'log dir {self.target_dir} does not exist')
        else:
            log_entry_names = os.listdir(self.target_dir)
            with ThreadPoolExecutor(max_workers=64) as pool:
                results = [
                    pool.submit(lambda x: (x, osp.exists(osp.join(self.target_dir, name, 'metadata.yaml'))), name) for
                    name in log_entry_names]
                pool.shutdown(wait=True)

            self.log_list = list(map(lambda x: x[0], filter(lambda x: x[1], [r.result() for r in results])))
            self._started = True
            return None

    def stop(self):
        self._reset()

    def _reset(self):
        self.pcd_frame_idx = 0
        self.rgb_frame_idx = 0
        self.pcd_sub_frame_idx = 0
        self.rgb_sub_frame_idx = 0
        self.log_list = []
        self._started = False

    def _pcd_next(self):
        if self.pcd_sub_frame_idx < len(self._stages) - 1:
            self.pcd_sub_frame_idx += 1
        else:
            self.pcd_sub_frame_idx = 0
            self.pcd_frame_idx += 1

    def capture_pcd(self) -> Optional[o3d.geometry.PointCloud]:
        logger.debug('FileCamera.capture_pcd()')
        if not self._started:
            raise Exception('camera is not started')

        if self.pcd_frame_idx >= len(self.log_list) or self.pcd_frame_idx < 0:
            return None

        if len(self._stages) * self.pcd_frame_idx + self.pcd_sub_frame_idx > len(
                self._stages) * self.rgb_frame_idx + self.rgb_sub_frame_idx:
            raise Exception(f'pcd frame index {self.pcd_frame_idx}.{self.pcd_sub_frame_idx} is larger than rgb frame index{self.rgb_frame_idx}.{self.rgb_sub_frame_idx}')

        pcd_path = self._pcd_path_fn[self._stages[self.pcd_sub_frame_idx]](self.log_list[self.pcd_frame_idx])
        if not osp.exists(pcd_path):
            self._pcd_next()
            self._rgb_next()
            raise FileNotFoundError(f'file {pcd_path} not found')
        else:
            # loading pcd
            _pcd = o3d.geometry.PointCloud()
            _npz_file = np.load(pcd_path)
            _pcd.points = o3d.utility.Vector3dVector(_npz_file['points'].astype(np.float32))
            _pcd.colors = o3d.utility.Vector3dVector(_npz_file['colors'].astype(np.float32) / 255.0)

            # calculating virtual pcd
            raw_virtual_pcd = _pcd

            camera_pcd = raw_virtual_pcd.transform(self.transforms.virtual_to_camera_transform)

            self._pcd_next()

            return camera_pcd

    def _rgb_next(self):
        if self.rgb_sub_frame_idx < len(self._stages) - 1:
            self.rgb_sub_frame_idx += 1
        else:
            self.rgb_sub_frame_idx = 0
            self.rgb_frame_idx += 1

    def capture_rgb(self) -> Optional[np.ndarray]:
        logger.debug('FileCamera.capture_rgb()')
        if not self._started:
            raise Exception('camera is not started')

        if self.rgb_frame_idx >= len(self.log_list) or self.rgb_frame_idx < 0:
            return None

        if len(self._stages) * self.rgb_frame_idx + self.rgb_sub_frame_idx > len(
                self._stages) * self.pcd_frame_idx + self.pcd_sub_frame_idx:
            raise Exception(f'rgb frame index {self.rgb_frame_idx}.{self.rgb_sub_frame_idx} is larger than pcd frame index {self.pcd_frame_idx}.{self.pcd_sub_frame_idx}')

        rgb_path = self._rgb_path_fn[self._stages[self.rgb_sub_frame_idx]](self.log_list[self.rgb_frame_idx])
        if not osp.exists(rgb_path):
            self._rgb_next()
            self._pcd_next()
            raise FileNotFoundError(f'file {rgb_path} not found')
        else:
            # get rgb
            rgb_img = cv2.imread(rgb_path)

            self._rgb_next()
            return rgb_img


if __name__ == '__main__':
    __FAKE_COLORS__ = [
        np.array([0.12156862745098039, 0.4666666666666667, 0.7058823529411765]),
        np.array([1.0, 0.4980392156862745, 0.054901960784313725]),
        np.array([0.17254901960784313, 0.6274509803921569, 0.17254901960784313]),
        np.array([0.8392156862745098, 0.15294117647058825, 0.1568627450980392]),
        np.array([0.5803921568627451, 0.403921568627451, 0.7411764705882353]),
        np.array([0.5490196078431373, 0.33725490196078434, 0.29411764705882354]),
        np.array([0.8901960784313725, 0.4666666666666667, 0.7607843137254902]),
        np.array([0.4980392156862745, 0.4980392156862745, 0.4980392156862745]),
        np.array([0.7372549019607844, 0.7411764705882353, 0.13333333333333333]),
        np.array([0.09019607843137255, 0.7450980392156863, 0.8117647058823529])
    ]


    def visualize_point_cloud_list_with_points(transformed_pcd_list, points: Union[np.ndarray, List[np.ndarray]] = None,
                                               fake_color=False, point_colors=None):
        coordinate = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        transformed_pcd_all = o3d.geometry.PointCloud()
        xyz_list = [np.asarray(pcd.points) for pcd in transformed_pcd_list]
        if fake_color:
            rgb_list = [np.tile(__FAKE_COLORS__[i], (xyz_list[i].shape[0], 1)) for i in range(len(xyz_list))]
        else:
            rgb_list = [np.asarray(pcd.colors) for pcd in transformed_pcd_list]
        transformed_pcd_all.points = o3d.utility.Vector3dVector(np.concatenate(xyz_list, axis=0))
        transformed_pcd_all.colors = o3d.utility.Vector3dVector(np.concatenate(rgb_list, axis=0))
        lookat = np.asarray(transformed_pcd_all.points).mean(axis=0).tolist()

        geometries = [coordinate, transformed_pcd_all]
        if points is not None and len(points) > 0 and all([isinstance(p, np.ndarray) for p in points]):
            for idx, p in enumerate(points):
                sphere = o3d.geometry.TriangleMesh().create_sphere(radius=0.02).translate(p)
                if point_colors is not None and len(point_colors) == len(points):
                    sphere.paint_uniform_color(__FAKE_COLORS__[point_colors[idx] % len(__FAKE_COLORS__)])
                else:
                    sphere.paint_uniform_color(__FAKE_COLORS__[idx % len(__FAKE_COLORS__)])
                geometries.append(sphere)

        o3d.visualization.draw_geometries(geometries,
                                          front=[0., 0., 1.],
                                          lookat=lookat,
                                          up=[0., 1., 0.],
                                          zoom=0.25)


    from planning.configs.config import config as planning_config

    experiment_option = omegaconf.OmegaConf.load('manifests/experiment_real_v1.yaml').experiment
    experiment_option.planning = OmegaConf.create(convert_dict(planning_config))
    experiment_option = config_completion(experiment_option)

    fcam = FileCamera(
        r'C:\Users\liyutong\Downloads\experiment_real',
        'data/calibration/v11/external_camera_calibration.txt',
        transforms=ExperimentRealTransforms(option=experiment_option)
    )
    fcam.start()

    _ = fcam.capture_rgb()
    _ = fcam.capture_pcd()
    _ = fcam.capture_pcd()
    _ = fcam.capture_rgb()

    while True:
        rgb = fcam.capture_rgb()
        pcd = fcam.capture_pcd()
        # rgb = fcam.capture_rgb()
        if rgb is None or pcd is None:
            break

        # visualize_point_cloud_list_with_points([pcd])
        # cv2.imshow('rgb', rgb)
        # cv2.waitKey(0)
