import os
import yaml
import json
import datetime
import git
import numpy as np
from typing import Union, Dict, List, Tuple, Optional, Any, Callable, Iterable, Sequence, Mapping, TypeVar, Generic
import threading
import open3d as o3d
import logging
import uuid
import cv2
from omegaconf import OmegaConf, DictConfig
"""
```yaml
logging:
  path: "./log"
```
"""
import os.path as osp


class Logger:
    """The experiment logger logs all the information of an experiment, including:
       - gripper movement
       - model predictions
       - point cloud
       - etc.

    This class is NOT thread safe
    """
    _metadata: dict = None
    # _lock: threading.Lock = None
    _inited: bool = False
    _closed: bool = False

    def __init__(self, namespace: str, config: Union[str, Dict[str, Any]], tag: str = None):
        self.namespace = namespace
        self.config = self._load_config(config) if isinstance(config, str) else config  # if config is a path, load it
        if tag is not None:
            self.tag = tag
        else:
            self.tag = 'default'
        self.identifier = str(uuid.uuid4())
        self.log_dir = osp.join(self.config['path'],
                                self.namespace,
                                self.tag,
                                self.identifier)
        # self._lock = threading.Lock()
        self._started_threads = []
        self._inited = False
        self._closed = False

    def init(self):
        self._init_log_dir()  # create log directory
        self._init_metadata()  # init metadata in memory
        self._inited = True

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, type, value, traceback):
        # on exit, save metadata
        self.close()

    def _load_config(self, config_path: str):
        """load config from yaml file
        """
        with open(config_path, 'r') as file:
            res = yaml.safe_load(file)
            assert 'logging' in res, 'logging config not found'
            res = res['logging']
            return res

    def _init_log_dir(self):
        """create log directory
        """
        os.makedirs(self.log_dir)
        os.makedirs(osp.join(self.log_dir, 'pcd'), exist_ok=True)
        os.makedirs(osp.join(self.log_dir, 'rgb'), exist_ok=True)
        os.makedirs(osp.join(self.log_dir, 'mask'), exist_ok=True)
        os.makedirs(osp.join(self.log_dir, 'particles'), exist_ok=True)
        os.makedirs(osp.join(self.log_dir, 'nocs'), exist_ok=True)

    def _init_metadata(self):
        """populate metadata with default values
        """
        metadata = {
            'uuid': str(uuid.uuid4()),
            'identifier': self.identifier,
            'tag': self.tag,
            'note': '',
            'garment_id': '',
            'commit': '',
            'model': None,
            'running_config': None,
            "checkpoint_name": '',
            'episode_idx': None,
            'trial_idx': None,
            'action': {
                'step': None,
                'type': ''
            },
            'pose_virtual': {
                'gripper': {
                },
                'prediction': {},
                'is_best': None
            },
            'pose_world': {
                'gripper': {
                },
                'prediction': {}
            },
            'prediction_info': {
                'reward': {
                    'virtual': {},
                    'real': {},
                },
                'reachable_matrix': {
                },
                'safe_to_pick_matrix': {
                }
            },
            'pcd': {
                'raw': {},
                'processed': {},
                'processed_nocs': {}
            },
            'calibration': {
                'camera_to_world': None,
                'robot_left_to_world': None,
                'robot_right_to_world': None
            },
            'decision': {}
        }

        self._metadata = metadata
        self._seq = []
        self._seq_length = None
        self.save()

    def log_running_config(self, config: Union[OmegaConf, DictConfig]):
        assert not self._closed, AssertionError("logger closed")
        if isinstance(config, DictConfig):
            config = OmegaConf.create(config)
        self._update_metadata('running_config', OmegaConf.to_container(config, resolve=True))
        
    def log_commit(self, path: str):
        """try to log the git commit of the current repo

        Args:
            path (str): path to repo
        """
        assert not self._closed, AssertionError("logger closed")
        try:
            repo = git.Repo(path).git
            commit = repo.rev_parse('HEAD', short=True)
            self._update_metadata('commit', commit)
        except Exception as e:
            logging.warning('Failed to log commit: {}'.format(e))
            pass

    def log_model(self, model_path: str, checkpoint_name: str = "last"):
        """log model configuration

        Args:
            model_path (str): path to model checkpoint directory
            checkpoint_name (str, optional): checkpoint name. Defaults to "last".
        """
        assert not self._closed, AssertionError("logger closed")
        with open(osp.join(model_path, 'config.yaml'), 'r') as f:
            model_config = yaml.load(f, Loader=yaml.SafeLoader)
        self._update_metadata('model', model_config)
        self._update_metadata('checkpoint_name', checkpoint_name)

    def log_garment_id(self, garment_id: str):
        """log garment id

        Args:
            garment_id (str): garment id
        """
        assert not self._closed, AssertionError("logger closed")
        self._update_metadata('garment_id', garment_id)

    def log_processed_file(self, flag: str):
        """
        Log .processed file for database processing

        Args:
            flag (str): flag to be written to .processed file
        """
        assert not self._closed, AssertionError("logger closed")
        processed_file = osp.join(self.log_dir, '.processed')
        with open(processed_file, 'w') as f:
            f.write(flag)

    def log_empty_annotation_file(self):
        """
        log empty annotation file for database processing
        """
        assert not self._closed, AssertionError("logger closed")
        annotation_file = osp.join(self.log_dir, 'annotation.yaml')
        with open(annotation_file, 'w') as f:
            f.write('')

    def log_calibration(self, camera_to_world: Union[str, Dict[str, Any], np.ndarray], robot_left_to_world: Union[str, Dict[str, Any], np.ndarray], robot_right_to_world: Union[str, Dict[str, Any], np.ndarray]):
        """log camera calibration

        Args:
            camera_to_world (Union[str, Dict[str, Any]]): camera to world transformation
            robot_left_to_world (Union[str, Dict[str, Any]]): robot left to world transformation
            robot_right_to_world (Union[str, Dict[str, Any]]): robot right to world transformation
        """
        assert not self._closed, AssertionError("logger closed")
        if isinstance(camera_to_world, str):
            with open(camera_to_world, 'r') as file:
                _camera_to_world = json.load(file)
        elif isinstance(camera_to_world, np.ndarray):
            _camera_to_world = camera_to_world.tolist()
        else:
            _camera_to_world = camera_to_world

        if isinstance(robot_left_to_world, str):
            with open(robot_left_to_world, 'r') as file:
                _robot_left_to_world = json.load(file)
        elif isinstance(robot_left_to_world, np.ndarray):
            _robot_left_to_world = robot_left_to_world.tolist()
        else:
            _robot_left_to_world = robot_left_to_world

        if isinstance(robot_right_to_world, str):
            with open(robot_right_to_world, 'r') as file:
                _robot_right_to_world = json.load(file)
        elif isinstance(robot_right_to_world, np.ndarray):
            _robot_right_to_world = robot_right_to_world.tolist()
        else:
            _robot_right_to_world = robot_right_to_world

        self._update_metadata('calibration.camera_to_world', _camera_to_world)
        self._update_metadata('calibration.robot_left_to_world', _robot_left_to_world)
        self._update_metadata('calibration.robot_right_to_world', _robot_right_to_world)

    def log_pose_gripper_virtual(self, key: str, left: np.ndarray, right: np.ndarray,
                                 is_best: bool = None):
        """log gripper pose in virtual space

        Args:
            key (str): begin, end, or whatever key
            left (np.ndarray): left pose, must bu 6d vector
            right (np.ndarray): right pose, must bu 6d vector
            is_best (bool): whether the current gripper pose is the best action pose
        """
        assert not self._closed, AssertionError("logger closed")
        assert isinstance(left, np.ndarray), 'pose must be numpy array'
        assert left.shape == (6, ), 'pose must be 6d vector (x, y, z, rx, ry, rz)'
        assert isinstance(right, np.ndarray), 'pose must be numpy array'
        assert right.shape == (6, ), 'pose must be 6d vector (x, y, z, rx, ry, rz)'
        self._update_metadata('pose_virtual.gripper.{}'.format(key), {
            'left': left.tolist(),
            'right': right.tolist()
        })
        if is_best is not None:
            self._update_metadata('pose_virtual.is_best', is_best)

    def log_pose_gripper_world(self, key: str, left: np.ndarray, right: np.ndarray):
        """log gripper pose in world space

        Args:
            key (str): begin, end, or whatever key
            left (np.ndarray): left pose, must bu 6d vector
            right (np.ndarray): right pose, must bu 6d vector
            seq (Optional[np.ndarray], optional): the sequenced position, must be 13d vector. Defaults to None.
        """
        assert not self._closed, AssertionError("logger closed")
        assert isinstance(left, np.ndarray), 'pose must be numpy array'
        assert left.shape == (6, ), 'pose must be 6d vector (x, y, z, rx, ry, rz)'
        assert isinstance(right, np.ndarray), 'pose must be numpy array'
        assert right.shape == (6, ), 'pose must be 6d vector (x, y, z, rx, ry, rz)'
        self._update_metadata('pose_world.gripper.' + key + '.left', left.tolist())
        self._update_metadata('pose_world.gripper.' + key + '.right', right.tolist())

            
    def log_pose_gripper_seq(self, seq: np.ndarray):
        """log gripper pose

        Args:
            key (str): begin, end, or whatever key
            left (np.ndarray): left pose, must bu 6d vector
            right (np.ndarray): right pose, must bu 6d vector
            seq (Optional[np.ndarray], optional): the sequenced position, must be 13d vector. Defaults to None.
        """
        assert not self._closed, AssertionError("logger closed")
        if self._seq_length is None:
            self._seq_length = len(seq)
        else:
            assert self._seq_length == len(seq), 'seq must be same length as previous seq'
        assert isinstance(seq, np.ndarray), 'seq must be numpy array'
        # assert seq.shape == (13, ), 'seq must be 13d vector (t, x1, y1, z1, rx1, ry1, rz1, x2, y2, z2, rx2, ry2, rz2)'
        self._seq.append(seq)  # TODO: check if this is thread safe


    def log_pose_prediction_virtual(self, key: str, pose: np.ndarray):
        """log predicted pose in virtual space

        Args:
            key (str): begin, end or whatever key
            pose (np.ndarray): pose, must be  n x 3d or n x 6d vector
        """
        assert not self._closed, AssertionError("logger closed")
        assert isinstance(pose, np.ndarray), 'pose must be numpy array'
        if len(pose.shape) == 2 and pose.shape[-1] == 3:
            # pad with zeros (r, p, y)
            pose = np.concatenate([pose, np.zeros((pose.shape[0], 3))], axis=-1)
        assert len(pose.shape) == 2 and pose.shape[-1] == 6, 'pose must be n x 6d vector (x, y, z, rx, ry, rz)'
        self._update_metadata('pose_virtual.prediction.' + key, pose.tolist())

    def log_pose_prediction_world(self, key: str, pose: np.ndarray):
        """log predicted pose in world space

        Args:
            key (str): begin, end or whatever key
            pose (np.ndarray): pose, must be  n x 3d or n x 6d vector
        """
        assert not self._closed, AssertionError("logger closed")
        assert isinstance(pose, np.ndarray), 'pose must be numpy array'
        if len(pose.shape) == 2 and pose.shape[-1] == 3:
            # pad with zeros (r, p, y)
            pose = np.concatenate([pose, np.zeros((pose.shape[0], 3))], axis=-1)
        assert len(pose.shape) == 2 and pose.shape[-1] == 6, 'pose must be n x 6d vector (x, y, z, rx, ry, rz)'
        self._update_metadata('pose_world.prediction.' + key, pose.tolist())

    def log_predicted_reward(self, key: str, reward: np.ndarray):
        """log predicted reward matrix for all predicted grasp points in fling action

        Args:
            key (str): virtual, real
            reward (np.ndarray): reward, must be  (n, n, k) matrix
        """
        assert not self._closed, AssertionError("logger closed")
        assert key in ('virtual', 'real')
        assert isinstance(reward, np.ndarray), 'reward must be numpy array'
        assert len(reward.shape) == 3
        self._update_metadata('prediction_info.reward.' + key, reward.tolist())

    def log_reachable_matrix(self, log_reachable_matrix: np.ndarray):
        """log log_reachable_matrix for all predicted grasp points in fling action

        Args:
            log_reachable_matrix (np.ndarray): reachable_matrix, must be (n, 2) ndarray with dtype np.bool
        """
        assert not self._closed, AssertionError("logger closed")
        assert isinstance(log_reachable_matrix, np.ndarray), 'reachable_matrix must be numpy array'
        assert len(log_reachable_matrix.shape) == 2
        self._update_metadata('prediction_info.reachable_matrix', log_reachable_matrix.tolist())

    def log_safe_to_pick_matrix(self, safe_to_pick_matrix: np.ndarray):
        """log safe_to_pick_matrix for all predicted grasp points in fling action

        Args:
            safe_to_pick_matrix (np.ndarray): safe_to_pick_matrix, must be (n, n) matrix with dtype np.bool
        """
        assert not self._closed, AssertionError("logger closed")
        assert isinstance(safe_to_pick_matrix, np.ndarray), 'safe_to_pick_matrix must be numpy array'
        assert len(safe_to_pick_matrix.shape) == 2
        self._update_metadata('prediction_info.safe_to_pick_matrix', safe_to_pick_matrix.tolist())

    def log_pcd_raw(self, key: str, pcd: o3d.geometry.PointCloud, only_npz: bool = False):
        """log raw point cloud

        Args:
            key (str): begin, end or whatever key
            pcd (o3d.geometry.PointCloud): point cloud
            only_npz (bool, optional): only log npz file. Defaults to False.
        """
        assert not self._closed, AssertionError("logger closed")
        self.log_pcd('raw', key, pcd, True, no_pcd=only_npz)

    def log_pcd_processed(self, key: str, pcd: o3d.geometry.PointCloud, only_npz: bool = False):
        """log processed point cloud

        Args:
            key (str): begin, end or whatever key
            pcd (o3d.geometry.PointCloud): point cloud
            only_npz (bool, optional): only log npz file. Defaults to False.
        """
        assert not self._closed, AssertionError("logger closed")
        self.log_pcd('processed', key, pcd, True, no_pcd=only_npz)
    
    def log_rgb(self, key: str, rgb_img: np.ndarray):
        """log rgb image

        Args:
            key (str): begin, end or whatever key
            rbg_img (np.ndarray): image from rgb camera
        """
        assert not self._closed, AssertionError("logger closed")
        if rgb_img is None or not isinstance(rgb_img, np.ndarray):
            return
        
        rgb_file = osp.join(self.log_dir, 'rgb', key + '.jpg')        
        t = threading.Thread(target=cv2.imwrite, args=(rgb_file, rgb_img[:, :, ::-1]))  # convert from RGB to BGR
        t.start()
        self._started_threads.append(t)
    
    def log_mask(self, key: str, mask_img: np.ndarray):
        """log mask image

        Args:
            key (str): begin, end or whatever key
            mask_img (np.ndarray): bool type array
        """
        assert not self._closed, AssertionError("logger closed")
        if mask_img is None or not isinstance(mask_img, np.ndarray):
            return
        
        mask_img = mask_img.astype(np.uint8) * 255
        mask_file = osp.join(self.log_dir, 'mask', key + '.png')
        t = threading.Thread(target=cv2.imwrite, args=(mask_file, mask_img))
        t.start()
        self._started_threads.append(t)

    @staticmethod
    def _log_pcd(pcd_file: str, pcd: o3d.geometry.PointCloud):
        o3d.io.write_point_cloud(pcd_file, pcd)

    @staticmethod
    def _log_pts(pts_file: str, pcd: o3d.geometry.PointCloud):
        points = np.asarray(pcd.points)
        colors =  np.asarray(pcd.colors)
        points = points.astype(np.half)
        colors = (colors * 255).astype(np.uint8)
        np.savez_compressed(pts_file, points=points, colors=colors)

    def log_pcd(self, type: str, key: str, pcd: o3d.geometry.PointCloud, create_npz: bool = False,
                no_pcd: bool = False):
        """where the actual logging happens

        Args:
            type (str): raw or proccessed
            key (str): begin, end or whatever key
            pcd (o3d.geometry.PointCloud): point cloud
            create_npz (bool, optional): whether to create npz file. Defaults to False.
            no_pcd (bool, optional): whether to omit logging pcd file. Defaults to False.
        """
        assert not self._closed, AssertionError("logger closed")
        pcd_file = osp.join(self.log_dir, 'pcd', type + '_' + key + '.ply')
        pcd_dest = osp.join('pcd', type + '_' + key)
        self._update_metadata('pcd.' + type + '.' + key, pcd_dest)
        # self._log_pcd(pcd_file, pcd)
        if not no_pcd:
            t = threading.Thread(target=self._log_pcd, args=(pcd_file, pcd))
            t.start()
            self._started_threads.append(t)

        if create_npz:
            pts_file = osp.join(self.log_dir, 'pcd', type + '_' + key + '.npz')
            t = threading.Thread(target=self._log_pts, args=(pts_file, pcd))
            t.start()
            self._started_threads.append(t)

    def log_particles(self, key: str, particles: np.ndarray):
        """
        log particles
        Args:
            key (str): begin, end or whatever key
            particles (np.ndarray): particles, must be (N, 3) shape
        """
        assert not self._closed, AssertionError("logger closed")
        assert isinstance(particles, np.ndarray), 'pose must be numpy array'
        assert len(particles.shape) == 2 and particles.shape[-1] == 3, 'pose must be n x 3d vector (x, y, z)'
        particle_file = osp.join(self.log_dir, 'particles', key + '.npy')
        particle_dest = osp.join('particles', key + '.npy')
        self._update_metadata('particles.' + key, particle_dest)
        t = threading.Thread(target=self._log_particles, args=(particle_file, particles))
        t.start()
        self._started_threads.append(t)

    def _log_particles(self, particles_file: str, particles: np.ndarray):
        np.save(particles_file, particles)

    def log_nocs(self, key: str, nocs: np.ndarray):
        """
        log nocs
        Args:
            key (str): begin, end or whatever key
            nocs (np.ndarray): nocs, must be (N, 3) shape
        """
        assert not self._closed, AssertionError("logger closed")
        assert isinstance(nocs, np.ndarray), 'pose must be numpy array'
        assert len(nocs.shape) == 2 and nocs.shape[-1] == 3, 'pose must be n x 3d vector (x, y, z)'
        nocs_file = osp.join(self.log_dir, 'nocs', key + '.npz')
        nocs_dest = osp.join('nocs', key + '.npz')
        self._update_metadata('pcd.processed_nocs.' + key, nocs_dest)
        t = threading.Thread(target=self._log_nocs, args=(nocs_file, nocs.astype(np.float16)))
        t.start()
        self._started_threads.append(t)

    def _log_nocs(self, nocs_file: str, nocs: np.ndarray):
        np.savez_compressed(nocs_file, nocs=nocs)

    def log_decision(self, key: str, decision: Iterable[int]):
        """log decision

        Args:
            key (str): begin, end or whatever key
            decision (Tuple[int]): decision, index of choosed prediction
        """
        assert not self._closed, AssertionError("logger closed")
        self._update_metadata('decision.' + key, list(decision))

    def log_reward(self, key: str, value_dict: dict):
        """
        log reward
        Args:
            key (str): begin, end or whatever key
            value_dict (dict): reward dict
        """
        assert not self._closed, AssertionError("logger closed")
        self._update_metadata('reward.' + key, value_dict)

    def log_note(self, note: str):
        """log note

        Args:
            note (str): note to be logged
        """
        assert not self._closed, AssertionError("logger closed")
        self._update_metadata('note', note)

    def log_episode_idx(self, episode_idx: int):
        """log episode index

        Args:
            episode_idx (int): episode index
        """
        assert not self._closed, AssertionError("logger closed")
        self._update_metadata('episode_idx', episode_idx)
    

    def log_trial_idx(self, trial_idx: int):
        """log trial index

        Args:
            trial_idx int: index of trial
        """
        assert not self._closed, AssertionError("logger closed")
        self._update_metadata('trial_idx', trial_idx)

    def log_action_step(self, step: int):
        """log step

        Args:
            step (int): step
        """
        assert not self._closed, AssertionError("logger closed")
        self._update_metadata('action.step', step)

    def log_action_type(self, action_type: str):
        """log action_type
        Args:
            action_type (str): action type
        """
        assert not self._closed, AssertionError("logger closed")
        self._update_metadata('action.type', action_type)

    def log_garment_id(self, garment_id: str):
        """log garment id

        Args:
            garment_id (str): garment id
        """
        assert not self._closed, AssertionError("logger closed")
        self._update_metadata('garment_id', garment_id)
        
    def _update_metadata(self, key="", value=None):
        """update in-memory metadata

        Args:
            key (str, optional): the path to attribute. Defaults to "".
            value (_type_, optional): attribute value. Defaults to None.
        """
        # TODO: check if this is thread safe
        if key != "":
            keys = key.split('.')
            current = self._metadata
            for k in keys[:-1]:
                if k not in current.keys():
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

    def save(self):
        """save metadata to file
        """
        with open(osp.join(self.log_dir, 'metadata.yaml'), 'w') as file:
            yaml.safe_dump({self.namespace: self._metadata}, file, sort_keys=False, default_flow_style=False)
            
        np.save(osp.join(self.log_dir, 'seq.npy'), np.array(self._seq))

    def finalize(self):
        self._update_metadata("_finalized", True)

    def close(self):
        if not self._closed and self._inited:
            self.save()
            for t in self._started_threads:
                t.join()
            self._closed = True

    def __del__(self):
        self.close()

if __name__ == '__main__':
    # test
    def generate_pseudo_pc(n=1000000):
        xyz = np.random.randint(-100, 100, size=(n, 3))
        colors = np.random.rand(n, 3)
        res = o3d.geometry.PointCloud()
        res.points = o3d.utility.Vector3dVector(xyz)
        res.colors = o3d.utility.Vector3dVector(colors)

        return res
    running_config = OmegaConf.create({'compat': {}})
    # running_config.compat.num_retries = 0
    
    # {'path':'./log'}
    with Logger('experiment_real', r'manifests/experiment_real_v1.yaml') as logger:
        # beginning of experiment
        logger.log_running_config(running_config)
        logger.log_commit('.')
        logger.log_model(r'outputs/outputs_action_v5_long_transformers')
        logger.log_calibration(r'data/calibration/world_to_camera_transform_v3.json', r'data/calibration/world_to_left_robot_transform_v3.json',
                               r'data/calibration/world_to_right_robot_trasnform_v3.json')
        logger.log_garment_id('test_garment_id')
        # before fling
        logger.log_pcd_raw('begin', generate_pseudo_pc())
        logger.log_pcd_processed('begin', generate_pseudo_pc())

        # after decision
        logger.log_pose_prediction_world('begin', np.random.rand(6, 6))
        logger.log_decision('begin', [0])

        # during action
        logger.log_pose_gripper_world('begin', np.random.rand(6, ), np.random.rand(6, ))  # (x, y, z, r, p, y)
        for _ in range(10):
            logger.log_pose_gripper_seq(seq=np.random.rand(13, ))  # (t, x1, y1, z1, rx1, ry1, rz1, x2, y2, z2, rx2, ry2, rz2)
        logger.log_pose_gripper_world('end', np.random.rand(6, ), np.random.rand(6, ))  # (x, y, z, r, p, y)

        # after action
        logger.log_pcd_raw('end', generate_pseudo_pc())
        logger.log_pcd_processed('end', generate_pseudo_pc())
        logger.log_note('this is a note')
