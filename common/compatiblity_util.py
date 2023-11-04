from typing import Dict, Any

def inference3D_predict_action_compat(new: Dict[str, Any]) -> Dict[str, Any]:
    old = dict(**new)
    if 'transform_idxs' in old.keys():
        del old['transform_idxs']
    
    if 'grasp_point_all' in old.keys():
        del old['grasp_point_all']
    
    if len(old['transforms']) == 0:
        old['transforms'] = None
    elif len(old['transforms']) == 1:
        old['transforms'] = old['transforms'][0]
    else:
        raise ValueError(f"cannot apply for {len(old['transforms'])} transforms")

    return old
