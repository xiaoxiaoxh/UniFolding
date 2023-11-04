import json
from pathlib import Path
from typing import List
import os
import os.path as osp

import cv2
import numpy as np

from common.image_util import segment
from manipulation.heuristics.planar_transform import PlanarTransform

_save_root_path: str = "./data/temp/"


def set_save_root_path(path: str):
    global _save_root_path
    _save_root_path = path
    if not osp.exists(_save_root_path):
        os.makedirs(_save_root_path, exist_ok=True)


def get_save_root_path() -> str:
    global _save_root_path
    return _save_root_path


class Instruction:
    instructions_path = Path('data') / 'instructions'
    # scale_factor = 0.7
    scale_factor = 0.8
    image_size = (1032, 772)  # (w, h)

    def __init__(self, folding_lines, grasp_points=None):
        self.folding_lines = folding_lines
        self.grasp_points = grasp_points

    @classmethod
    def crop_image_for_template(cls, image: np.ndarray, template: np.ndarray, vis: bool = False):
        """
        Crop and resize the image according to the size of the template image

        @params: image, (W, H, 3) numpy ndarray
        @params: template, (W', H') numpy ndarray
        """
        template_h, template_w = template.shape[:2]
        raw_h, raw_w = image.shape[:2]
        template_aspect_ratio = template_w / template_h
        crop_w = int(raw_h * template_aspect_ratio)
        center_w = raw_w // 2
        start_w = max(0, center_w - crop_w // 2)
        end_w = min(raw_w - 1, center_w + crop_w // 2)
        crop_image = image[:, start_w:end_w]
        resized_image = cv2.resize(crop_image, (template_w, template_h
                                                ), cv2.INTER_NEAREST)
        if vis:
            cv2.imshow('crop_resized_image', resized_image)
        return resized_image

    @classmethod
    def affine_transform(cls, image: np.ndarray, vis: bool = False):
        center = [image.shape[1] / 2, image.shape[0] / 2]
        mat = cv2.getRotationMatrix2D(center, 0., cls.scale_factor)
        color = (image[0, 0] + image[-1, -1] + image[0, -1] + image[-1, 0]) / 4
        if vis:
            cv2.imshow('raw_template', image)
        transformed_image = cv2.warpAffine(image, mat, (image.shape[1], image.shape[0]),
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=color)
        if vis:
            cv2.imshow('transformed_template', transformed_image)
        return transformed_image

    @classmethod
    def load_instructions(cls, name: str) -> List:
        center = np.array([cls.image_size[0] / 2, cls.image_size[1] / 2])
        with open(str(cls.instructions_path / f'{name}.json')) as f:
            data = json.load(f)
            instructions = []
            for instr in data['instructions']:
                folding_lines = (np.asarray(instr['folding-lines']) - center[np.newaxis, np.newaxis, :]) \
                                * cls.scale_factor + center[np.newaxis, np.newaxis, :]
                grasp_points = (np.asarray(instr['grasp-points']) - center[np.newaxis, :]) \
                               * cls.scale_factor + center[np.newaxis, :]
                instruction = Instruction(folding_lines, grasp_points)
                instructions.append(instruction)
            return instructions

    @classmethod
    def get_template(cls, name: str):
        template = cv2.imread(str(cls.instructions_path / f'{name}-template.png'), cv2.IMREAD_GRAYSCALE)
        template = cls.affine_transform(template, vis=False)
        return template

    def draw(self, image):
        for i, (start, end) in enumerate(self.folding_lines):
            l1 = np.array(start)
            l2 = np.array(end)
            cv2.line(image, l1.astype(int), l2.astype(int), (0, 241, 255), 5)
            cv2.putText(image, f'{i}', (l1 + [2, -10]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 241, 255), 2, cv2.LINE_AA)

        if self.grasp_points is not None:
            cv2.circle(image, np.array(self.grasp_points[0]).astype(int), 2, (0, 0, 0), 2)
            cv2.circle(image, np.array(self.grasp_points[1]).astype(int), 2, (0, 0, 0), 2)


class TemplateMatching:
    """TemplateMatching used in SpeedFolding"""

    def __init__(self):
        self.rng = np.random.default_rng()

    @staticmethod
    def rotate_image(image, theta: float, offset=(0, 0)):
        center = np.array(image.shape[1::-1]) / 2
        t = cv2.getRotationMatrix2D(center, -theta * 180 / np.pi, 1.0)
        t[0, 2] += offset[0]
        t[1, 2] += offset[1]
        return cv2.warpAffine(image, t, image.shape[1::-1])

    def get_matched_instruction(self, mask, template_name: str, save=False, image=None):
        # Make sure that the center of mass of the template's segmentation mask is precisely at the center of the image        
        template = Instruction.get_template(template_name)

        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        template_color = cv2.cvtColor(template, cv2.COLOR_GRAY2RGB)

        # Initital guess based on mask
        _, mask_info = segment(mask_color)
        _, template_info = segment(template_color)

        x = (mask_info['x'] - template_info['x']) * template.shape[1]
        y = (mask_info['y'] - template_info['y']) * template.shape[0]

        def score_pose(pose):
            return np.mean((mask == self.rotate_image(template, pose[2], pose[:2])).astype(np.float32))

        # Particle based template matching
        best_pose, best_score = [], 0.0
        for theta in np.linspace(0.0, 2 * np.pi, 60):
            new_pose = np.array([x, y, theta])
            score = score_pose(new_pose)
            if score > best_score:
                best_score = score
                best_pose = new_pose

        for i in range(12):
            std = np.array([40.0, 40.0, 0.4]) / (i + 1)
            new_poses = self.rng.normal(best_pose, std, size=(20, 3))
            for new_pose in new_poses:
                score = score_pose(new_pose)
                if score > best_score:
                    best_score = score
                    best_pose = new_pose

        # Transform lines into new frame
        center = np.array(mask.shape[1::-1]) / 2
        offset = best_pose[:2] + center
        rot_mat = PlanarTransform.get_rotation_matrix(-best_pose[2])

        instructions = Instruction.load_instructions(template_name)

        for instr in instructions:
            for j in range(len(instr.folding_lines)):
                instr.folding_lines[j][0] = rot_mat @ (instr.folding_lines[j][0] - center) + offset
                instr.folding_lines[j][1] = rot_mat @ (instr.folding_lines[j][1] - center) + offset

            if instr.grasp_points is not None:
                instr.grasp_points[0] = rot_mat @ (instr.grasp_points[0] - center) + offset
                instr.grasp_points[1] = rot_mat @ (instr.grasp_points[1] - center) + offset

        # Rank possible instructions according to y position of first line
        instruction = sorted(instructions, key=lambda x: x.folding_lines[0][0][1] + x.folding_lines[0][1][1], reverse=True)[0]

        if save:
            image_draw = image if image is not None else mask
            image_draw = cv2.cvtColor(image_draw, cv2.COLOR_GRAY2RGB)

            if image is not None:
                fitted_mask = self.rotate_image(template, best_pose[2], best_pose[:2])
                image_draw[:, :, 2] = cv2.addWeighted(image_draw[:, :, 2], 0.75, mask, 0.25, 0)
                image_draw[:, :, 1] = cv2.addWeighted(image_draw[:, :, 1], 0.75, fitted_mask, 0.25, 0)

            instruction.draw(image_draw)

            cv2.imshow('matching', image_draw)
            cv2.waitKey()
            cv2.imwrite(osp.join(get_save_root_path(), 'output_template_matching.png'), image_draw)

        return instruction


if __name__ == '__main__':
    import os.path as osp
    import os

    # image_root_dir = './log/captures/tshirt_short_canonical_corl'
    image_root_dir = './log/captures/tshirt_long_canonical_corl'
    for image_name in os.listdir(image_root_dir):
        image_dir = osp.join(image_root_dir, image_name)

        image_path = osp.join(image_dir, 'rgb', 'begin.jpg')
        mask_path = osp.join(image_dir, 'mask', 'begin.png')
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        template = Instruction.get_template('shirt')

        image_crop = Instruction.crop_image_for_template(image, template, vis=False)
        image_crop_gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        mask_crop = Instruction.crop_image_for_template(mask, template, vis=False)[:, :, 0]

        m = TemplateMatching()

        instruction = m.get_matched_instruction(mask_crop, template_name='shirt', save=True, image=image_crop_gray)
