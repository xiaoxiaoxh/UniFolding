import os
import os.path as osp
import json
import numpy as np

class CLOTH3DObjLoader(object):
    "a tool class for iteratively return CLOTH3D .obj path"

    def __init__(self,
                 data_root: str,
                 sub_category_json_path: str,
                 meta_json_path: str,
                 category: str = 'Tshirt',
                 sub_categories: tuple = (2,),
                 use_wsl: bool = True,
                 num_splits: int = 1,
                 seed: int = 224,
                 instance_filter_file_path: str = None):
        self.data_root = data_root
        self.category = category
        self.use_wsl = use_wsl
        self.num_splits = num_splits
        assert osp.exists(data_root) and osp.exists(sub_category_json_path) and osp.exists(meta_json_path)
        # get category metadata
        with open(meta_json_path, 'r') as f:
            self.category_meta_dict = json.load(f)
        self.category_scale = self.category_meta_dict['category_scale']
        self.category_center = self.category_meta_dict['category_center']
        # load sub category dict
        with open(sub_category_json_path, 'r') as f:
            sub_cate_dict = json.load(f)
        # filter sub categories
        self.instance_ids = []
        for id, sub_category in sub_cate_dict.items():
            if sub_category in sub_categories:
                self.instance_ids.append(id)
        self.instance_ids = sorted(self.instance_ids)

        # filter instance ids according to pre-defined file
        self.instance_filter_file_path = instance_filter_file_path
        if self.instance_filter_file_path is not None:
            assert osp.exists(instance_filter_file_path), 'instance_filter_file_path does not exist!'
            with open(instance_filter_file_path, 'r') as f:
                valid_instance_ids = f.readlines()
                valid_instance_ids = [instance_id.strip() for instance_id in valid_instance_ids]
            self.instance_ids = [instance_id for instance_id in self.instance_ids
                                 if instance_id in valid_instance_ids]
        # create obj path list from instance ids
        self._obj_path_list = []
        for instance_id in self.instance_ids:
             self._obj_path_list.append(self.get_obj_path(instance_id))
        # create splits and generate a list of sub-lists
        self._split_obj_path_list = self.split_list(self._obj_path_list, self.num_splits)

        self.seed = seed
        np.random.seed(self.seed)

    def shuffle_instances(self):
        np.random.shuffle(self._obj_path_list)
        # create splits and generate a list of sub-lists
        self._split_obj_path_list = self.split_list(self._obj_path_list, self.num_splits)

    def get_obj_path(self, instance_id: str) -> str:
        obj_path = osp.join(self.data_root, instance_id,
                            '{}.obj'.format(self.category))
        assert osp.exists(obj_path), f'{obj_path} does not exist!'
        if self.use_wsl:
            assert obj_path.startswith('/mnt/c/'), f'{obj_path} is not a WSL path!'
            obj_path = obj_path.replace('/mnt/c/', 'C:/')
        return obj_path

    @staticmethod
    def split_list(lst, n):
        """
        Split a list into n chunks
        """
        avg = len(lst) / float(n)
        out = []
        last = 0.0

        while last < len(lst):
            out.append(lst[int(last):int(last + avg)])
            last += avg

        return out

    @property
    def all_obj_path_list(self):
        return self._obj_path_list

    @property
    def split_obj_path_list(self):
        return self._split_obj_path_list

if __name__ == '__main__':
    obj_loader = CLOTH3DObjLoader(data_root='/mnt/c/Users/xuehan/Data/CLOTH3D/train',
                                  sub_category_json_path='/mnt/c/Users/xuehan/Data/CLOTH3D/train_meta/Tshirt_sub_categories_v3.json',
                                  meta_json_path='/mnt/c/Users/xuehan/Data/CLOTH3D/train_meta/Tshirt_meta.json',
                                  category='Tshirt',
                                  sub_categories=(2, ),
                                  num_splits=4,
                                  use_wsl=True,
                                  instance_filter_file_path='/home/hanxue/data/UniFolding/VR_Imitation/vr_imitation_flattening-folding-long_v5_dataset/Tshirt/train_instances.txt')
    for obj_path in obj_loader.all_obj_path_list:
        print(obj_path)

    for split_obj_path_list in obj_loader.split_obj_path_list:
        print(len(split_obj_path_list))
        for obj_path in split_obj_path_list:
            print(obj_path)
        print('======================')
