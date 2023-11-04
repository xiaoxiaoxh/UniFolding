import numpy as np
import skimage.transform as st
import scipy.interpolate as si
import scipy.ndimage as sni
import itertools
from learning.cloth_funnels.geometry import get_center_affine, pixel_to_3d, transform_points, transform_pose, get_pointcloud
def camera_image_to_view(cam_img, tx_camera_view, img_shape=(128,128)):
    tf_camera_view = st.AffineTransform(matrix=tx_camera_view)
    result_img = st.warp(cam_img, tf_camera_view.inverse, output_shape=img_shape)
    return result_img

class ImageStackTransformer:
    """
    Note: this class follows skimage.transform coordinate convention
    (x,y) x right, y down
    The rest of skimage uses (col, row) convention
    """
    def __init__(self, img_shape=(128,128), 
            rotations=np.linspace(-np.pi, np.pi, 17), 
            scales=[1.0, 1.5, 2.0, 2.5, 3.0]):
        """
        Create a stack of rotations * scales images.
        Rotation: counter-clockwise
        Scales: >1 object appear bigger
        """
        assert len(img_shape) == 2
        stack_shape = (len(rotations) * len(scales),) + tuple(img_shape)

        transforms = list()
        self.transform_tuples = list(itertools.product(rotations, scales))

        for rot, scale in itertools.product(rotations, scales):
            # both skimage and torchvision use
            tf = get_center_affine(
                img_shape=img_shape, 
                rotation=rot, scale=scale)
            tf.params = tf.params.astype(np.float32)
            transforms.append(tf)

        self.shape = stack_shape
        self.transforms = transforms
        self.rotations = rotations
        self.scales = scales
    
    def forward_img(self, img, mode='constant'):
        results = [st.warp(img, tf.inverse, mode=mode, preserve_range=True) for tf in self.transforms]
        stack = np.stack(results).astype(np.uint8)
        return stack
    
    def forward_raw(self, raw, tx_camera_view):
        img_shape = self.shape[1:]
        stack = np.empty(
            (len(self.transforms),) + img_shape + raw.shape[2:], 
            dtype=raw.dtype)
        for i, tf in enumerate(self.transforms):
            ntf = st.AffineTransform(tf.params @ tx_camera_view)
            stack[i] = st.warp(raw, ntf.inverse, 
                order=1,
                output_shape=img_shape,
                preserve_range=True)
        return stack
    
    def inverse_coord(self, stack_coord):
        """
        Convert 3d stack coordinate integers to
        float coordinate in the original image
        """
        return self.transforms[stack_coord[0]].inverse(stack_coord[1:])

    def get_inverse_coord_map(self):
        identity_map = np.moveaxis(
            np.indices(self.shape[1:], dtype=np.float32)[::-1],0,-1
            )

        maps = list()
        for tf in self.transforms:
            tx = np.linalg.inv(tf.params)
            r = transform_points(
                identity_map.reshape(-1,2), 
                tx).reshape(identity_map.shape)
            maps.append(r)
        coord_stack = np.stack(maps)
        return coord_stack

    def get_world_coords_stack(self, depth, tx_camera_view, tx_world_camera, cam_intr):
        img_coords_stack = self.get_inverse_coord_map()
        raw_img_coords_stack = transform_points(
            img_coords_stack.reshape(-1,2), 
            np.linalg.inv(tx_camera_view)).reshape(
                img_coords_stack.shape)

        # x,y
        # transform to world coord
        world_coords_stack = np.empty(
            img_coords_stack.shape[:-1]+(3,), 
            dtype=np.float32)
        for i in range(len(img_coords_stack)):
            img_coords = raw_img_coords_stack[i]
            # skimage uses (x,y) coordinate, pixel_to_3d uses (y,x)
            coords_3d = pixel_to_3d(depth, img_coords.reshape(-1,2)[:,::-1], 
                cam_pose=tx_world_camera, cam_intr=cam_intr)
            img_coords_3d = coords_3d.reshape(img_coords.shape[:-1] + (3,))
            world_coords_stack[i] = img_coords_3d
        return world_coords_stack


def is_coord_valid_robot(coords, tx_robot_world, 
        reach_radius=0.93, near_radius=0.0755):
    """
                    max     recommended             
    reach_radius:   0.946   0.85
    near_radius:    0       0.0755

    Reference:
    https://www.universal-robots.com/articles/ur/application-installation/what-is-a-singularity/
    """
    coords_robot = transform_points(coords, tx_robot_world)
    dist_3d = np.linalg.norm(coords_robot, axis=-1)
    dist_xy = np.linalg.norm(coords_robot[...,:2], axis=-1)
    is_valid = (dist_3d < reach_radius) & (dist_xy > near_radius)
    return is_valid

def is_coord_valid_table(coords, table_low=(-0.58,-0.88,-0.05), table_high=(0.58,0.87,0.2)):
    is_valid = np.ones(coords.shape[:-1], dtype=bool)
    for i in range(3):
        this_valid = (table_low[i] < coords[...,i]) & (coords[...,i] < table_high[i])
        is_valid = is_valid & this_valid
    return is_valid


def fill_nearest(depth_im, mask):
    coords = np.moveaxis(np.indices(depth_im.shape),0,-1)
    interp = si.NearestNDInterpolator(coords[~mask], depth_im[~mask])
    out_im = depth_im.copy()
    out_im[mask] = interp(coords[mask])
    return out_im

def get_offset_stack(stack, offset=16):
    """
    Assuming (N,H,W,D)
    up: move up offset pixels
    down: move down offset pixels
    """
    value = np.nan
    if stack.dtype is np.dtype('bool'):
        value = False
    up_stack = np.full(stack.shape, value, dtype=stack.dtype)
    down_stack = np.full(stack.shape, value, dtype=stack.dtype)
    up_stack[:,offset:,...] = stack[:,:-offset,...]
    down_stack[:,:-offset:,...] = stack[:,offset:,...]
    return up_stack, down_stack


def check_line_validity(stack, offset=16, axis=1, eps=1e-7):
    length = offset*2+1
    weights = np.full((length,),1/length, dtype=np.float32)
    result = sni.convolve1d(stack.astype(np.float32), 
        weights, axis=axis, mode='constant', cval=0)
    out = result > (1-eps)
    return out
