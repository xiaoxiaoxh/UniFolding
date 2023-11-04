import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from learning.cloth_funnels.keypoint_detector.networks.keypoint_deeplab import KeypointDeeplab
from learning.cloth_funnels.keypoint_detector.common.torch_util import to_numpy

class KeypointDetector():
    def __init__(self, model_path, input_size=128):

        self.model_path = model_path
        self.network = KeypointDeeplab.load_from_checkpoint(model_path).float()
        self.network.eval()

    def get_keypoints(self, image, mask=None, model_dim=128):
        """
        Takes in NxNx3 image, returns Nx3 keypoints
        """
        assert image.shape[-1] == 3

        original_dims = image.shape[-2:]

        image = cv2.resize(image, (model_dim, model_dim))

        image = (image - np.array([0.5, 0.5, 0.5]))/np.array([0.5, 0.5, 0.5])
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, 0)
        image = torch.tensor(image).float()
        image = image.to(self.network.device)

        with torch.no_grad():
            out = self.network.forward(image)

        scoremap = to_numpy(out['scoremap'][0])
        mask_downsampled = cv2.resize(mask, scoremap.shape[1:3])
        mask_expanded = np.repeat(np.expand_dims(mask_downsampled, 0), scoremap.shape[0], axis=0)
        scoremap = np.where(mask_expanded == 0, -np.inf, scoremap)
        scoremap_max_idx = np.argmax(scoremap.reshape(scoremap.shape[0], -1), axis=-1)
        scoremap_max_coords_np = np.stack(np.unravel_index(scoremap_max_idx, shape=scoremap.shape[-2:])).T[:, ::-1].astype(np.float64)
        scoremap_max_coords_np *= (original_dims[0] / model_dim)

        return scoremap_max_coords_np
