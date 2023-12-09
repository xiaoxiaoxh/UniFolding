import os
import os.path as osp
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import omegaconf
import open3d as o3d
import plyer
import py_cli_interaction
import requests
import tqdm
import yaml
from omegaconf import OmegaConf, DictConfig

from common.datamodels import AnnotationConfig, AnnotationFlag
from tools.data_management.src.common.datamodels import AnnotationPoint, LogPoint

from functools import reduce

class AnnotatorIO(ABC):
    def __init__(self, opt: AnnotationConfig):
        self.opt = opt

    @abstractmethod
    def get_pcd(self, entry_name: str) -> Tuple[Optional[o3d.geometry.PointCloud], Optional[Exception]]:
        raise NotImplementedError

    @abstractmethod
    def get_rgb(self, entry_name: str) -> Tuple[Optional[np.ndarray], Optional[Exception]]:
        raise NotImplementedError

    @abstractmethod
    def get_raw_log(self, entry_name: str) -> Tuple[Optional[omegaconf.DictConfig], Optional[Exception]]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def select_log_dir(cls, gui: Optional[bool] = None) -> Tuple[str, Optional[Exception]]:
        raise NotImplementedError

    @abstractmethod
    def scan_log_dir(self) -> Tuple[List[str], List[str], List[str], Optional[Exception]]:
        raise NotImplementedError

    @abstractmethod
    def set_log_processed_flag(self, entry_name: str, processed: AnnotationFlag) -> Optional[Exception]:
        raise NotImplementedError

    @abstractmethod
    def get_log_processed_flag(self, entry_name: str) -> Tuple[int, Optional[Exception]]:
        raise NotImplementedError

    @abstractmethod
    def acquire_annotation_lock(self, entry_name: str) -> Optional[Exception]:
        raise NotImplementedError

    @abstractmethod
    def release_annotation_lock(self, entry_name: str) -> Optional[Exception]:
        raise NotImplementedError

    @abstractmethod
    def move_for_backup(self, entry_name: str, timeout_s=5) -> Optional[Exception]:
        raise NotImplementedError

    @abstractmethod
    def write_annotation(
            self,
            entry_name: str,
            annotation_dict: Dict[str, Any],
            annotated_log: DictConfig,
    ) -> Optional[Exception]:
        raise NotImplementedError


class AnnotatorDiskIO(AnnotatorIO):

    def __init__(self, opt: AnnotationConfig):
        super().__init__(opt)

    def get_pcd(self, entry_name: str) -> Tuple[Optional[o3d.geometry.PointCloud], Optional[Exception]]:
        value = osp.join(self.opt.root_dir, entry_name, "pcd", "raw_begin.npz")
        if isinstance(value, str) and '.ply' in value:
            return o3d.io.read_point_cloud(value), None
        elif isinstance(value, str) and '.npz' in value:
            _curr_pcd = o3d.geometry.PointCloud()
            npz_file = np.load(value)
            pc_xyz = npz_file['points'].astype(np.float32)
            pc_rgb = npz_file['colors'].astype(np.float32) / 255.0
            _curr_pcd.points = o3d.utility.Vector3dVector(pc_xyz)
            _curr_pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
            return _curr_pcd, None
        elif isinstance(value, o3d.geometry.PointCloud):
            return value, None
        else:
            return o3d.geometry.PointCloud(), Exception(f'unknown type {type(value)}')

    def get_rgb(self, entry_name: str) -> Tuple[Optional[np.ndarray], Optional[Exception]]:
        value = osp.join(self.opt.root_dir, entry_name, "rgb", "begin.jpg")
        if isinstance(value, str):
            return cv2.imread(value), None
        elif isinstance(value, np.ndarray):
            return value, None
        else:
            return np.empty([]), Exception(f'unknown type {type(value)}')

    def get_raw_log(self, entry_name: str) -> Tuple[Optional[omegaconf.DictConfig], Optional[Exception]]:
        value = osp.join(self.opt.root_dir, entry_name, "metadata.yaml")
        if isinstance(value, str):
            obj = OmegaConf.load(value)
            return obj, None
        elif isinstance(value, dict):
            obj = OmegaConf.create(value)
            return obj, None
        elif isinstance(value, omegaconf.DictConfig):
            return value, None
        else:
            return OmegaConf.create({}), Exception(f'unknown type {type(value)}')

    @classmethod
    def select_log_dir(cls, gui: Optional[bool] = None) -> Tuple[str, Optional[Exception]]:
        """select log dir via gui

        Returns:
            Tuple[str, Optional[Exception]]: path of selected log dir and exception if any
        """
        if gui is not None and not gui:
            res = py_cli_interaction.must_parse_cli_string('please input log dir',
                                                           osp.realpath(osp.join(os.getcwd(), 'logs')))
            return res, None
        else:
            try:
                sel = plyer.filechooser.choose_dir(title='please select log dir', path=osp.realpath(os.getcwd()))
            except Exception as e:
                res = py_cli_interaction.must_parse_cli_string('[failed to open gui] please input log dir',
                                                               osp.realpath(osp.join(os.getcwd(), 'logs')))
                return res, None

            if sel is None or len(sel) == 0:
                return '', None

            res = sel[0]
            return res, None

    def scan_log_dir(self) -> Tuple[List[str], List[str], List[str], Optional[Exception]]:
        """scan log dir and return all log files

        Returns:
            Tuple[List[str], Optional[Exception]]: result as:
            - list of unprocessed log entry names
            - list of processed log entry names
            - exception if any
        """
        if self.opt.root_dir == "" or self.opt.root_dir is None or not osp.exists(self.opt.root_dir):
            return [], [], [], Exception(f'log dir {self.opt.root_dir} does not exist')
        else:
            log_entry_names = os.listdir(self.opt.root_dir)
            with ThreadPoolExecutor(max_workers=64) as pool:
                results = [pool.submit(lambda x: (x, self.get_log_processed_flag(x)), name) for name in log_entry_names]
                for r in tqdm.tqdm(futures.as_completed(results), total=len(results)):
                    if r.result()[1][1] is not None:
                        pool.shutdown(wait=False)
                        return [], [], [], r.result()[1][1]
                pool.shutdown(wait=True)

            results = [r.result() for r in results]
            unprocessed_log_entries = list(map(lambda x: x[0], filter(lambda x: x[1][0] == 0, results)))
            processed_log_entries = list(map(lambda x: x[0], filter(lambda x: x[1][0] == 1, results)))
            corrupted_log_entries = list(map(lambda x: x[0], filter(lambda x: x[1][0] == 2, results)))

            return unprocessed_log_entries, processed_log_entries, corrupted_log_entries, None

    def set_log_processed_flag(self, entry_name: str, processed: AnnotationFlag) -> Optional[Exception]:
        """set log processed flag

        Args:
            entry_name (str): name of log entry
            processed (bool): processed flag

        Returns:
            Optional[Exception]: exception if any
        """
        try:
            with open(osp.join(self.opt.root_dir, entry_name, '.processed'), 'w') as f:
                f.write(str(processed.value))
            return None
        except Exception as e:
            return e

    def get_log_processed_flag(self, entry_name: str) -> Tuple[int, Optional[Exception]]:
        """get log processed flag

        Args:
            entry_name (str): name of log entry

        Returns:
            Tuple[bool, Optional[Exception]]: processed flag and exception if any
        """
        try:
            if not osp.exists(osp.join(self.opt.root_dir, entry_name, '.processed')):
                return 0, None
            with open(osp.join(self.opt.root_dir, entry_name, '.processed'), 'r') as f:
                content = f.read()
                if content == '':
                    return 0, None
                else:
                    return int(content), None
        except Exception as e:
            return -1, e

    def acquire_annotation_lock(self, entry_name: str) -> Optional[Exception]:
        """acquire the lock to annotate

        Args:
            entry_name (str): name of log entry

        Returns:
            Optional[Exception]: exception if any
        """
        try:
            if osp.exists(osp.join(self.opt.root_dir, entry_name, '.lock')):
                return Exception('failed to acquire lock')
            unique_id = str(uuid.uuid4())
            with open(osp.join(self.opt.root_dir, entry_name, '.lock'), 'w') as f:
                f.write(unique_id)
            time.sleep(0.1)
            with open(osp.join(self.opt.root_dir, entry_name, '.lock'), 'r') as f:
                content = f.read()

            if content != unique_id:
                return Exception('failed to acquire lock')
            else:
                return None
        except Exception as e:
            return e

    def release_annotation_lock(self, entry_name: str) -> Optional[Exception]:
        """release the lock to annotate

        Args:
            entry_name (str): name of log entry

        Returns:
            Optional[Exception]: exception if any
        """
        try:
            if osp.exists(osp.join(self.opt.root_dir, entry_name, '.lock')):
                os.remove(osp.join(self.opt.root_dir, entry_name, '.lock'))
                return None
            else:
                return Exception('lock not acquired')
        except Exception as e:
            return e

    def move_for_backup(self, entry_name: str, timeout_s=5) -> Optional[Exception]:
        """move the file to file.bak, check if file.bak exists, if it exits, move file.bak to file.{i}.back

        Args:
            entry_name: name of entry
            timeout_s: timeout in seconds
        """
        path = osp.join(self.opt.root_dir, entry_name, "metadata.yaml")
        _name = osp.basename(path)
        _dirname = osp.dirname(path)

        _bak_path = osp.join(_dirname, f"{_name}.bak")
        if osp.exists(_bak_path):
            _index = 1
            _start_t = 1
            while True:
                if not osp.exists(_another_bak_path := osp.join(_dirname, f"{_name}.{_index}.bak")):
                    shutil.move(_bak_path, _another_bak_path)
                    break
                if time.time() - _start_t > timeout_s:
                    return Exception(TimeoutError)
        shutil.move(path, _bak_path)
        return None

    def write_annotation(
            self,
            entry_name: str,
            annotation_dict: Dict[str, Any],
            annotated_log: DictConfig
    ) -> Optional[Exception]:
        """write annotation
        Args:
            entry_name (str): name of log entry
            annotation_dict: annotation dict (partial)
            annotated_log: annotated log (full)

        Returns:
            Tuple[AnnotationContext, Optional[Exception]]: annotation context (not used) and exception if any
        """

        try:
            with open(osp.join(self.opt.root_dir, entry_name, "annotation.yaml"), "w") as f:
                yaml.dump({self.opt.raw_log_namespace: {"annotation": annotation_dict}}, f)

            with open(osp.join(self.opt.root_dir, entry_name, "metadata.yaml"), "w") as f:
                OmegaConf.save(annotated_log, f)

            return None
        except Exception as e:
            return e


class AnnotatorNetworkIO(AnnotatorIO):
    def __init__(self, opt: AnnotationConfig):
        super().__init__(opt)
        self.session = requests.Session()
        self.query_filter = {}

    def get_pcd(self, entry_name: str) -> Tuple[Optional[o3d.geometry.PointCloud], Optional[Exception]]:
        url = f"{self.opt.api_url}/v1/proxy/{entry_name}"
        rel_path = "pcd/raw_begin.npz"
        try:
            resp = self.session.get(url, params={'rel_path': rel_path})
            resp.raise_for_status()

            # Load the npz file content
            npz_data = np.load(BytesIO(resp.content))

            # Convert to Open3D point cloud
            _curr_pcd = o3d.geometry.PointCloud()
            pc_xyz = npz_data['points'].astype(np.float32)
            pc_rgb = npz_data['colors'].astype(np.float32) / 255.0
            _curr_pcd.points = o3d.utility.Vector3dVector(pc_xyz)
            _curr_pcd.colors = o3d.utility.Vector3dVector(pc_rgb)

            return _curr_pcd, None
        except Exception as e:
            return None, e

    def get_rgb(self, entry_name: str, img_name: str = "begin") -> Tuple[Optional[np.ndarray], Optional[Exception]]:
        url = f"{self.opt.api_url}/v1/proxy/{entry_name}"
        rel_path = f"rgb/{img_name}.jpg"
        try:
            resp = self.session.get(url, params={'rel_path': rel_path})
            resp.raise_for_status()

            nparr = np.frombuffer(resp.content, np.uint8)  # Convert the content  to a numpy array of bytes
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode the image as BGR (OpenCV's default color format)
            # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Convert the image from BGR to RGB

            return img_bgr, None
        except Exception as e:
            return None, e

    def get_raw_log(self, entry_name: str) -> Tuple[Optional[omegaconf.DictConfig], Optional[Exception]]:
        url = f"{self.opt.api_url}/v1/proxy/{entry_name}"
        rel_path = "metadata.yaml"
        try:
            resp = self.session.get(url, params={'rel_path': rel_path})
            resp.raise_for_status()

            # Assuming the response content is the raw YAML text
            yaml_str = resp.text
            config = OmegaConf.create(yaml_str)

            return config, None
        except Exception as e:
            return None, e
        
    def get_annotation(self, entry_name: str) -> Tuple[Optional[dict], Optional[Exception]]:
        url = f"{self.opt.api_url}/v1/logs"
        resp = self.session.get(url, json={'identifiers': [entry_name], 'extra_filter': None})
        if resp.status_code != 200:
            return {}, Exception(resp.json()['msg'])
        else:
            return resp.json()['logs'][0]['annotations'], None

    @classmethod
    def select_log_dir(cls, gui: Optional[bool] = None) -> Tuple[str, Optional[Exception]]:
        res = py_cli_interaction.must_parse_cli_string('please input tag:', 'default')
        return res, None

    def scan_log_dir(self) -> Tuple[List[str], List[str], List[str], Optional[Exception]]:
        unp_extra_filter = {
            '$and': [
                self.opt.extra_filter if self.opt.extra_filter is not None else {},
                {
                    '$or': [
                        {
                            '_processed': {
                                '$exists': False
                            }
                        },
                        {
                            '_processed': {
                                '$eq': 0
                            }
                        }
                    ]
                },
                {
                    "metadata.experiment_real": {
                        "$exists": True
                    }
                }
            ]
        }
        p_extra_filter = {
            '$and': [
                self.opt.extra_filter if self.opt.extra_filter is not None else {},
                {
                    '$and': [
                        {
                            '_processed': {
                                '$exists': True
                            }
                        },
                        {
                            '_processed': {
                                '$eq': 1
                            }
                        }
                    ]
                }
            ]
        }

        url = f"{self.opt.api_url}/v1/logs"
        unp_resp = self.session.post(url, json={'identifiers': None, 'extra_filter': unp_extra_filter})
        p_resp = self.session.post(url, json={'identifiers': None, 'extra_filter': p_extra_filter})
        if unp_resp.status_code == 200:
            unprocessed_logs = [res['identifier'] for res in unp_resp.json()['logs']]
        else:
            return [], [], [], Exception(unp_resp.json()['msg'])
        if p_resp.status_code == 200:
            # processed_logs = [zip(res['annotations'].keys(), [res['identifier']]* len(res['annotations'].keys()) ) \
            #                         for res in p_resp.json()['logs']]
            # processed_logs = reduce(lambda x, y: list(x) + list(y), processed_logs)
            processed_logs = [(list(res['annotations'].keys())[0], res['identifier']) for res in p_resp.json()['logs']]
        else:
            return [], [], [], Exception(p_resp.json()['msg'])
        return unprocessed_logs, processed_logs, [], None

    def set_log_processed_flag(self, entry_name: str, processed: AnnotationFlag) -> Optional[Exception]:
        url = f"{self.opt.api_url}/v1/processed/{entry_name}"
        resp = self.session.put(url, params={'value': int(processed.value)})
        if resp.status_code == 200:
            return None
        else:
            return Exception(resp.json()['msg'])

    def get_log_processed_flag(self, entry_name: str) -> Tuple[int, Optional[Exception]]:
        url = f"{self.opt.api_url}/v1/processed/{entry_name}"
        resp = self.session.get(url)
        if resp.status_code == 200:
            return int(resp.json()['value']), None
        else:
            return -1, Exception(resp.json()['msg'])

    def acquire_annotation_lock(self, entry_name: str) -> Optional[Exception]:
        url = f"{self.opt.api_url}/v1/locks/{entry_name}"
        resp = self.session.post(url)
        if resp.status_code == 200:
            return None
        else:
            return Exception(resp.json()['msg'])

    def release_annotation_lock(self, entry_name: str) -> Optional[Exception]:
        url = f"{self.opt.api_url}/v1/locks/{entry_name}"
        resp = self.session.delete(url)
        if resp.status_code == 200:
            return None
        else:
            return Exception(resp.json()['msg'])

    def move_for_backup(self, entry_name: str, timeout_s=5) -> Optional[Exception]:
        pass

    def write_annotation(
            self,
            entry_name: str,
            annotation_dict: Dict[str, Any],
            annotated_log: DictConfig,
    ) -> Optional[Exception]:
        url = f"{self.opt.api_url}/v1/logs"
        resp = self.session.get(url, json={'identifiers': [entry_name], 'extra_filter': None})
        if resp.status_code != 200:
            return Exception(resp.json()['msg'])
        log_point = LogPoint().from_dict(resp.json()['logs'][0])
        log_point.metadata_bak.append(log_point.metadata)
        log_point.metadata = OmegaConf.to_container(annotated_log, resolve=True)
        log_point.annotators.append(self.opt.annotator) if self.opt.annotator not in log_point.annotators else None
        idx = int(time.time() * 1000)
        log_point.annotations[f"{self.opt.annotator}.{idx}"] = annotation_dict

        url = f"{self.opt.api_url}/v1/logs/upload"
        resp = self.session.post(url, json=log_point.to_dict())
        if resp.status_code != 200:
            return Exception(resp.json()['msg'])
        else:
            return None


_disk_io_module = None
_net_io_module = None


def _get_disk_io_module(opt: AnnotationConfig):
    global _disk_io_module
    if _disk_io_module is None:
        _disk_io_module = AnnotatorDiskIO(opt)

    return _disk_io_module


def _get_net_io_module(opt: AnnotationConfig):
    global _net_io_module
    if _net_io_module is None:
        _net_io_module = AnnotatorNetworkIO(opt)

    return _net_io_module


def get_io_module(opt: AnnotationConfig) -> AnnotatorIO:
    return _get_net_io_module(opt)
