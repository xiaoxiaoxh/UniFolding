import os.path as osp
from typing import List

class ModelPathLoader:
    """Find all model checkpoint paths given multiple directories"""
    def __init__(self, model_root_list: List[str],
                        model_dir_list: List[List[str]]):
        self._model_path_list = []
        assert len(model_root_list) == len(model_dir_list)
        for idx, model_dir in enumerate(model_root_list):
            cur_model_work_dir_list = sorted(model_dir_list[idx])
            model_name = 'last.ckpt'
            for work_dir in cur_model_work_dir_list:
                model_path = osp.join(model_dir, work_dir, 'checkpoints', model_name)
                assert osp.exists(model_path)
                self._model_path_list.append(osp.join(model_dir, work_dir))

    def get_model_path_list(self):
        return self._model_path_list
