import copy
import itertools
import os.path as osp
import shutil
from typing import List, Optional, Tuple

import numpy as np
import omegaconf
import py_cli_interaction
import tqdm
import yaml
from omegaconf import OmegaConf

from common.datamodels import AnnotationConfig, AnnotationContext, ActionTypeDef
from tools.run_annotation.functionals import (
    visualize_point_cloud_list_with_points,
    pick_n_points_from_pcd,
)
from tools.run_annotation.io import get_io_module


def do_init_annotation(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """run init annotation

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context and exception if any

    Init the annotation context with the log entry.
    """
    context.entry_name = entry_name
    context.annotation_result.annotator = opt.annotator

    # use the side effect and verify the log entry
    _x = list(map(lambda x: list(x), context.raw_log[opt.raw_log_namespace].pose_virtual.prediction.begin))
    return context, None


def do_action_type_annotation(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """run action type annotation

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context and exception if any

    The annotation software displays the raw (un-masked) point cloud (before the action) with RGB. Then the user types the correct action type (e.g. fling, pick-and-place or fold1).
    """
    __DISPLAY_OPTIONS__ = [
        "甩动（fling）",
        "抓和放（pick-and-place）",
        "折叠（fold1）",
        "跳过（skip）"
    ]
    __DISPLAY_OPTIONS_MAPPING__ = [
        ActionTypeDef.FLING,
        ActionTypeDef.PICK_AND_PLACE,
        ActionTypeDef.FOLD_1,
        ActionTypeDef.DONE,
    ]

    try:
        while True:
            context.console.print("预测的动作（predicted action）: " + str(context.raw_log[opt.raw_log_namespace].action),
                                  style="yellow")
            context.console.print("【指令】在弹出的3D窗口中观察，判断下一步动作应该属于以下那种类型：", __DISPLAY_OPTIONS__,
                                  "然后按下Esc关闭窗口\n")
            visualize_point_cloud_list_with_points([context.curr_pcd])

            type_sel = py_cli_interaction.must_parse_cli_sel("选择正确的动作类型（select action type）",
                                                             __DISPLAY_OPTIONS__)
            res = __DISPLAY_OPTIONS_MAPPING__[type_sel]

            context.console.print("你的选择是（your selection is）:", res, style="blue")
            confirm = py_cli_interaction.must_parse_cli_bool("确认（confirm）?", default_value=True)
            if confirm:
                context.annotation_result.action_type = res
                break
            else:
                continue
    except KeyboardInterrupt as e:
        context.console.print("keyboard interrrupt", style="red")
        return context, Exception(KeyboardInterrupt)

    return context, None


def do_action_pose_annotation(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """run action pose annotation

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context and exception if any

    The annotation software displays the raw (un-masked) point cloud (before the action) with RGB and two pair of possible grasp-points predicted by the policy model. The user will click on the dense point cloud to annotate the correct pick points or place points for this action.
    """
    res = [None] * 4
    try:
        while True:
            if context.annotation_result.action_type in [ActionTypeDef.FOLD_1]:
                context.console.print(
                    "【指令】选择理想的抓点，按顺序点击左抓、右抓、左放、右放四个点，然后按下Esc关闭窗口\nselect ideal poses, click 4 points: left_pick, right_pick, left_place, right_place")
                pts, _, err = pick_n_points_from_pcd(context.curr_pcd, 4)
                if err is not None:
                    context.console.print(err, style="red")
                    continue
                res[0], res[1], res[2], res[3] = pts
            elif context.annotation_result.action_type in [ActionTypeDef.PICK_AND_PLACE]:
                context.console.print(
                    "【指令】选择理想的抓点，按顺序点击左抓、右抓、左放、右放四个点，然后按下Esc关闭窗口\nselect ideal poses, click 4 points: left_pick, right_pick, left_place, right_place")
                pts, _, err = pick_n_points_from_pcd(context.curr_pcd, 4)
                if err is not None:
                    context.console.print(err, style="red")
                    continue
                res[0], res[1], res[2], res[3] = pts
            elif context.annotation_result.action_type in [ActionTypeDef.FLING]:
                context.console.print(
                    "【指令】选择理想的抓点，按顺序点击左抓、右抓两个点，然后按下Esc关闭窗口\nselect ideal poses, click 2 points: left_pick, right_pick")
                pts, _, err = pick_n_points_from_pcd(context.curr_pcd, 2)
                if err is not None:
                    context.console.print(err, style="red")
                    continue
                res[0], res[1] = pts

            elif context.annotation_result.action_type in [ActionTypeDef.DONE]:
                return context, None

            else:
                return context, Exception(NotImplementedError)

            visualize_point_cloud_list_with_points([context.curr_pcd], points=pts)
            context.console.print("你的选择是（your selection is）:", res, style="blue")
            confirm = py_cli_interaction.must_parse_cli_bool("确认（confirm）?", default_value=True)
            if confirm:
                context.annotation_result.action_poses = list(
                    map(
                        lambda x: np.array([x[0], x[1], x[2], 0, 0, 0]) if x is not None else np.zeros(shape=(6,),
                                                                                                       dtype=float),
                        res
                    )
                )
                break
            else:
                continue
    except KeyboardInterrupt as e:
        context.console.print("keyboard interrrupt", style="red")
        return context, Exception(KeyboardInterrupt)

    return context, None


def do_grasp_ranking_annotation(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """run grasp ranking annotation

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context and exception if any

    Grasp-point Ranking Annotation: This step is only for fling action!!! The annotation software will repeat the following steps for K times:

    The annotation software displays the raw (un-masked) point cloud (before the action) with RGB and two pair of possible grasp-points predicted by the policy model. We denote the the first pair of points as P1, and the second pair of points as P2.
    The volunteer gives 4 possible ranking annotation:

    P1 > P2 (P1 is better)
    P1 < P2 (P2 is better)
    P1 = P2 (equally good)
    Not comparable (hard to distinguish for humans).
    """

    __DISPLAY_OPTIONS__ = [
        "左图和右图差不多（P1 = P2）",
        "左图比右图好（P1 > P2）",
        "右图比左图好（P1 < P2）",
        "很混乱，比不出来（Not comparable）"
    ]
    __DISPLAY_OPTIONS_MAPPING__ = [
        2,
        0,
        1,
        3
    ]
    __DISPLAY_OPTIONS_MAPPING_INV__ = [
        1,
        2,
        0,
        3
    ]
    __DISPLAY_POINTCLOUD_OFFSET__ = np.array([
        1.5, 0., 0.
    ])
    __NOT_COMPARABLE_INDEX__ = 3
    __REWARD_THRESH_INIT__ = -0.2
    __REWARD_THRESH_FINAL__ = -0.5
    __REWARD_NEGATIVE_INF__ = -1000.0
    __USE_DYNAMIC_THRESH__ = True
    __REWARD_TOP_RATIO__ = 0.2
    __MIN_NUM_CANDIDATES_PAIR__ = 5
    __ALWAYS_SELECT_TOP_CANDIDATES__ = True
    __MIN_NUM_CANDIDATES_PAIR_PRE_FILTERING__ = 10

    def is_grasp_point_safe(points: np.ndarray):
        d = np.linalg.norm(points[0][:3] - points[1][:3])
        return float(d) > 0.15  # TODO: remove magic number

    def is_grasp_point_unique(s, c, points):
        ret = True
        for index in s:
            curr_points = np.array([c[x] for x in index[:2]])
            for i in range(2):
                for j in range(2):
                    if 1e-4 < np.linalg.norm(points[i][:3] - curr_points[j][:3]) < 0.02:  # TODO: remove magic number
                        ret = False
            if not ret:
                break
        return ret

    candidates = list(map(lambda x: list(x), context.raw_log[opt.raw_log_namespace].pose_virtual.prediction.begin))

    if __USE_DYNAMIC_THRESH__:
        reward_matrix = np.array(context.raw_log[opt.raw_log_namespace].prediction_info.reward.virtual).mean(axis=-1)
        assert len(reward_matrix.shape) == 2
        num_candidates = reward_matrix.shape[0]  # K
        # make sure that the same point is not selected
        self_idxs = np.arange(num_candidates)
        reward_matrix[self_idxs, self_idxs] = __REWARD_NEGATIVE_INF__
        flatten_pair_score = reward_matrix.reshape((-1,))  # (K*K, )
        sorted_pair_idxs = np.argsort(flatten_pair_score)[::-1]  # (K*K, )
        thresh_pair_idx = sorted_pair_idxs[int(len(flatten_pair_score) * __REWARD_TOP_RATIO__)]
        idx1 = thresh_pair_idx // num_candidates
        idx2 = thresh_pair_idx % num_candidates
        _reward_thresh = reward_matrix[idx1, idx2]
    else:
        _reward_thresh = __REWARD_THRESH_INIT__

    while True:
        candidates_pair = list(zip(*np.where(np.triu(
            np.array(context.raw_log[opt.raw_log_namespace].prediction_info.reward.virtual).squeeze() - _reward_thresh,
            k=1) >= 0)))
        # list(itertools.combinations(range(0, len(candidates)), 2))
        if len(candidates_pair) < __MIN_NUM_CANDIDATES_PAIR__ and _reward_thresh > __REWARD_THRESH_FINAL__:
            _reward_thresh -= 0.1
        else:
            break
    if len(candidates_pair) < __MIN_NUM_CANDIDATES_PAIR__:
        context.console.print("太多坏的抓点，回到传统算法（too many bad grasp points, fallback to traditional algorithm）",
                              style="red")
        all_candidates_pair = list(itertools.combinations(range(0, len(candidates)), 2))
        if __ALWAYS_SELECT_TOP_CANDIDATES__:
            res_num = __MIN_NUM_CANDIDATES_PAIR_PRE_FILTERING__ - len(candidates_pair)
            np.random.shuffle(all_candidates_pair)
            candidates_pair.extend(all_candidates_pair[:res_num])
        else:
            candidates_pair = all_candidates_pair

    candidates_pair = list(map(lambda x: (int(x[0]), int(x[1])), candidates_pair))
    combos = list(itertools.combinations(candidates_pair, 2))
    if opt.K > len(combos):
        return context, Exception(f"trying to run K comparison with {len(combos)} is illegal")

    combo_indices = np.random.permutation(np.arange(0, len(combos)))

    if context.annotation_result == ActionTypeDef.DONE:
        return context, None
    elif context.annotation_result.action_type != ActionTypeDef.FLING:
        selected_grasp_point_indices: List[Optional[List[int]]] = [[0, 0, 0, 0] for _ in range(opt.K)]
        grasp_point_rankings = [__NOT_COMPARABLE_INDEX__] * opt.K
    else:
        while True:
            selected_grasp_point_indices: List[Optional[List[
                int]]] = []  # np.array(combos)[selected_indices_non_result].reshape(len(selected_indices_non_result), -1)
            grasp_point_rankings: List[Optional[int]] = []
            with tqdm.tqdm(total=opt.K) as pbar:
                for ranking_idx, compare_idx in enumerate(combo_indices):
                    left_op, right_op = combos[compare_idx]
                    left_points_np, right_points_np = np.array([candidates[x] for x in left_op]), np.array(
                        [candidates[x] for x in right_op])

                    if len(combo_indices) - ranking_idx + pbar.n > opt.K:
                        if not (is_grasp_point_safe(left_points_np) and is_grasp_point_safe(right_points_np)):
                            continue

                        if not (
                                is_grasp_point_unique(
                                    selected_grasp_point_indices,
                                    candidates,
                                    left_points_np
                                ) and
                                is_grasp_point_unique(
                                    selected_grasp_point_indices,
                                    candidates,
                                    right_points_np
                                )
                        ):
                            continue
                    else:
                        context.console.print("无效的数量（insufficient grasp pairs detected, disable filtering）",
                                              style="red")

                    context.console.print("\n抓点对（grasp_point_pair）: ", combos[compare_idx], style="yellow")
                    # Generate copy of point cloud
                    left_pcd, right_pcd = copy.deepcopy(context.curr_pcd), copy.deepcopy(context.curr_pcd)

                    # Move the point cloud for a distance
                    right_pcd = right_pcd.translate(__DISPLAY_POINTCLOUD_OFFSET__)
                    right_points_np[:, :3] += __DISPLAY_POINTCLOUD_OFFSET__

                    all_points = np.vstack([left_points_np[..., :3], right_points_np[..., :3]])
                    context.console.print(
                        "【指令】请观察并比较左（P1）右（P2）两幅图（please observe and compare P1(left) to P2(right)），哪个抓点最好")
                    visualize_point_cloud_list_with_points([left_pcd, right_pcd], points=all_points)

                    _sel = py_cli_interaction.must_parse_cli_sel("排序（ranking）: ", __DISPLAY_OPTIONS__)
                    _res = __DISPLAY_OPTIONS_MAPPING__[_sel]

                    selected_grasp_point_indices.append(list(left_op) + list(right_op))
                    grasp_point_rankings.append(_res)
                    pbar.update()

                    if pbar.n >= opt.K:
                        break

            context.console.print("你的选择是（your selection is）:",
                                  [__DISPLAY_OPTIONS_MAPPING_INV__[x] for x in grasp_point_rankings], style="blue")
            confirm = py_cli_interaction.must_parse_cli_bool("确认（confirm）?", default_value=True)
            if confirm:
                break
            else:
                continue

    context.annotation_result.grasp_point_rankings = grasp_point_rankings
    context.annotation_result.selected_grasp_point_indices = selected_grasp_point_indices

    return context, None


def do_fling_automatic_annotation(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """do fling automatic annotation

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context as completed

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context (not used) and exception if any

    We need to add Step 4 for fling action.
    In Step 2, the GT pick points from human annotator could be used for oracle ranking, because they are probably better than any predicted grasp-points from the AI model. So we need to perform Step 4 after Step 3:

    The annotation software displays two identical masked point cloud (before the action) with RGB, then shows GT pick points on the left side, and all other predicted pick points on the right side.
    The user should give 2 possible annotation:
    Case 1: The GT pick points are better than any other predicted point pairs.
    Case 2: The GT pick points are not comparable with other predicted point pairs (usually happpens under very crumpled garment states with multiple good candidates).
    Append the GT pick points into virtual_posses.predictions of the metadata.
    Append the automatic ranking annotation into annotation.selected_grasp_point_indices and annotation.grasp_point_rankings.
    Case 1: All the additional ranking results are P1 > P2 (label 0).
    Case 2: All the additional ranking results are P1 not comparbale with P2 (label 3).
    """
    __DISPLAY_POINTCLOUD_OFFSET__ = np.array([
        1.5, 0., 0.
    ])
    if context.annotation_result.action_type == ActionTypeDef.FLING:
        # Generate copy of point cloud
        left_pcd, right_pcd = copy.deepcopy(context.curr_pcd), copy.deepcopy(context.curr_pcd)
        left_points_np = np.array(context.annotation_result.action_poses[:2])
        right_points_np = np.array(list(
            map(lambda x: np.array(list(x)), context.raw_log[opt.raw_log_namespace].pose_virtual.prediction.begin)))

        # Move the point cloud for a distance
        right_pcd = right_pcd.translate(__DISPLAY_POINTCLOUD_OFFSET__)
        right_points_np[:, :3] += __DISPLAY_POINTCLOUD_OFFSET__

        all_points = np.vstack([left_points_np[..., :3], right_points_np[..., :3]])

        context.console.print("【指令】请观察并判断左图是不是比右图的每一种组合更好")
        visualize_point_cloud_list_with_points([left_pcd, right_pcd], points=all_points)

        context.annotation_result.fling_gt_is_better_than_rest = py_cli_interaction.must_parse_cli_bool(
            "左图是不是比右图的每一种组合更好（is GT better than any other predicted pairs）?")
        context.console.print("你的选择是（your selection is）: ", context.annotation_result.fling_gt_is_better_than_rest,
                              style="blue")

    else:
        context.annotation_result.fling_gt_is_better_than_rest = None

    return context, None


def do_finalize(opt: AnnotationConfig, entry_name: str, context: AnnotationContext) -> Tuple[
    AnnotationContext, Optional[Exception]]:
    """do finalize

    Args:
        opt (AnnotationConfig): options
        entry_name (str): name of log entry
        context (AnnotationContext): annotation context as completed

    Returns:
        Tuple[AnnotationContext, Optional[Exception]]: annotation context (not used) and exception if any
    """
    annotated_log = copy.deepcopy(context.raw_log)

    with omegaconf.open_dict(annotated_log):
        __FLING_GT_IS_BETTER_THAN_REST_MAPPING__ = {True: 0, False: 1, None: 3}
        _k = len(annotated_log[opt.raw_log_namespace].pose_virtual.prediction.begin)
        candidates_pair = list(itertools.combinations(range(0, _k), 2))

        if context.annotation_result.fling_gt_is_better_than_rest is not None:
            # Insert points to data
            annotated_log[opt.raw_log_namespace].pose_virtual.prediction.begin += list(
                map(lambda x: omegaconf.ListConfig(x.tolist()), context.annotation_result.action_poses[:2]))
            __k = list(range(_k, len(annotated_log[opt.raw_log_namespace].pose_virtual.prediction.begin)))

            # add generated result
            context.annotation_result.selected_grasp_point_indices.extend([[*__k, *x] for x in candidates_pair])
            context.annotation_result.grasp_point_rankings.extend([__FLING_GT_IS_BETTER_THAN_REST_MAPPING__[
                                                                       context.annotation_result.fling_gt_is_better_than_rest]] * len(
                candidates_pair))

        else:
            annotated_log[opt.raw_log_namespace].pose_virtual.prediction.begin += list(
                map(lambda x: omegaconf.ListConfig(x.tolist()), np.zeros(shape=(2, 6))))

            context.annotation_result.selected_grasp_point_indices.extend([[0, 0, 0, 0] for _ in candidates_pair])
            context.annotation_result.grasp_point_rankings.extend([__FLING_GT_IS_BETTER_THAN_REST_MAPPING__[
                                                                       context.annotation_result.fling_gt_is_better_than_rest]] * len(
                candidates_pair))

        annotation_dict = context.annotation_result.to_dict()
        annotated_log[opt.raw_log_namespace].annotation = omegaconf.DictConfig(annotation_dict)

    err = get_io_module(opt).move_for_backup(entry_name)
    if err is not None:
        context.console.print("失败（failed）", style="red")
        return context, err

    err = get_io_module(opt).write_annotation(entry_name, annotation_dict, annotated_log)
    if err is not None:
        context.console.print("失败（failed）", style="red")
        return context, err

    return context, None
