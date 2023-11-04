import cv2
import numpy as np

from common.image_util import segment
from manipulation.heuristics.planar_transform import PlanarTransform
from manipulation.heuristics.utils import find_first_point_in_contour, \
    get_normal_theta_at_contour_index, maximize_area_of_polygon_along_contour

_save_root_path: str = "./data/temp/"


def set_save_root_path(path: str):
    global _save_root_path
    _save_root_path = path
    if not os.path.exists(_save_root_path):
        os.makedirs(_save_root_path, exist_ok=True)


def get_save_root_path() -> str:
    global _save_root_path
    return _save_root_path


class Line:
    def __init__(self, start, end):
        self.start = np.array(start)
        self.end = np.array(end)

    def to_absolute_pixels(self, image_shape):
        return Line(
            start=[image_shape[1] * self.start[0], image_shape[0] * self.start[1]],
            end=[image_shape[1] * self.end[0], image_shape[0] * self.end[1]]
        )


class FoldingHeuristic:
    """FoldingHeuristic from SpeedFolding"""

    def __init__(self):
        # self.mask_path = [(110, 84), (1900, 104), (1899, 1180), (118, 1196)]
        self.mask_path = [(0, 57),(1032, 110), (1032, 741), (0, 741)]

    @staticmethod
    def getPerpCoord(a, b, length):
        v = (b - a) / np.linalg.norm(b - a)
        temp = v[0]
        v[0] = 0 - v[1]
        v[1] = temp
        return b + v * length

    @staticmethod
    def mirrorPoint(a, b, p):
        diff = np.array([a[1] - b[1], b[0] - a[0]])
        D = 2 * (diff[0] * (p[0] - a[0]) + diff[1] * (p[1] - a[1])) / np.linalg.norm(diff) ** 2
        return np.array([p[0] - diff[0] * D, p[1] - diff[1] * D])

    def calculate(self, image, line, same_gripper_distance=30.0, save=False):
        new_line = line.to_absolute_pixels(image.shape)

        img_thresh, _, all_contour = segment(image, return_contour=True)

        c = self.getPerpCoord(new_line.start, new_line.end, 800)
        d = self.getPerpCoord(new_line.end, new_line.start, -800)

        pts = np.array([[new_line.start, new_line.end, c, d]])
        cv2.fillPoly(img_thresh, pts.astype(int), 0)
        contours2, _ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        areas = [cv2.contourArea(c) for c in contours2]
        second_max_index = np.argsort(areas, axis=0)[-1]
        secondary_contour = contours2[second_max_index]

        c1 = find_first_point_in_contour(new_line.start, new_line.end, all_contour)
        c2 = find_first_point_in_contour(new_line.end, new_line.start, all_contour)

        # cv.drawContours(image, [all_contour], 0, (0, 255, 0), 1)
        # cv.circle(image, c1.astype(int), 4, (255, 255, 255), -1)
        # cv.circle(image, c2.astype(int), 4, (255, 255, 255), -1)

        mask_mirrored_path = [tuple(self.mirrorPoint(new_line.start, new_line.end, p).astype(int)) for p in self.mask_path]
        (point_left_idx, point_right_idx) = maximize_area_of_polygon_along_contour(secondary_contour, c1, c2, mask_path=(self.mask_path, mask_mirrored_path))

        pick_left = PlanarTransform(
            position=secondary_contour[point_left_idx][0],
            theta=get_normal_theta_at_contour_index(point_left_idx, secondary_contour),
        )

        pick_right = PlanarTransform(
            position=secondary_contour[point_right_idx][0],
            theta=get_normal_theta_at_contour_index(point_right_idx, secondary_contour),
        )

        place_left = pick_left.mirror_along_line(new_line)
        place_right = pick_right.mirror_along_line(new_line)

        # Currently set place orientation to pick orientation
        place_left.theta = pick_left.theta
        place_right.theta = pick_right.theta

        # Check if two points are nearby
        if np.linalg.norm(pick_left.position - pick_right.position, 2) < same_gripper_distance:
            pick_left.position = (pick_left.position + pick_right.position) / 2
            pick_left.theta = (pick_left.theta + pick_right.theta) / 2
            pick_right = None
            place_right = None

        if save:
            img = np.copy(image)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            self.draw_mask(img)
            self.draw_plan(img, line, (pick_left, pick_right, place_left, place_right))
            cv2.imshow('folding_poses_output', img)
            cv2.waitKey()
            cv2.imwrite(osp.join(get_save_root_path(), 'folding_poses_output.png'), img)

        return pick_left, pick_right, place_left, place_right

    def draw_mask(self, image):
        pts = np.array(self.mask_path, np.int32).reshape((1, -1, 2))
        cv2.polylines(image, pts, True, (0, 255, 255), 1)

    def draw_plan(self, image, line, points):
        new_line = line.to_absolute_pixels(image.shape)

        pick_left, pick_right, place_left, place_right = points

        cv2.line(image, new_line.start.astype(int), new_line.end.astype(int), (0, 120, 255), 3)

        # Draw pick and place points
        pick_left.draw(image)
        place_left.draw(image)

        if pick_right is not None:
            pick_right.draw(image)
            place_right.draw(image)


if __name__ == '__main__':
    import os.path as osp
    import os
    from manipulation.heuristics.matching import Instruction, TemplateMatching
    from manipulation.heuristics.matching import set_save_root_path as set_matching_save_root_path

    # image_dir = './log/experiment_real/tshirt_short_action14_real_corl/2023-08-05T13-23-58-13c9d0af'
    image_root = './log/captures/tshirt_short_canonical_corl'
    # image_root = './log/captures/tshirt_long_canonical_corl'

    for log_name in os.listdir(image_root):
        set_matching_save_root_path(f'./log/visualization/folding/{log_name}')
        image_dir = osp.join(image_root, log_name)
        image_path = osp.join(image_dir, 'rgb', 'begin.jpg')
        mask_path = osp.join(image_dir, 'mask', 'begin.png')
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        template = Instruction.get_template('shirt')

        image_crop = Instruction.crop_image_for_template(image, template, vis=False)
        image_crop_gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        mask_crop = Instruction.crop_image_for_template(mask, template, vis=False)[:, :, 0]

        m = TemplateMatching()
        # get folding lines from particle-swarm optimization
        instruction = m.get_matched_instruction(mask_crop, template_name='shirt', save=True, image=image_crop_gray)

        # line = Line(start=[0.58, 0.1], end=[0.57, 0.9])

        fh = FoldingHeuristic()
        for i, l in enumerate(instruction.folding_lines):
            set_save_root_path(f'./log/visualization/folding/{log_name}/{i}')
            is_last = (i == len(instruction.folding_lines) - 1)
            line = Line(start=[l[0][0] / image_crop_gray.shape[1], l[0][1] / image_crop_gray.shape[0]],
                        end=[l[1][0] / image_crop_gray.shape[1], l[1][1] / image_crop_gray.shape[0]])

            pick1, pick2, place1, place2 = fh.calculate(mask_crop, line, save=True)
