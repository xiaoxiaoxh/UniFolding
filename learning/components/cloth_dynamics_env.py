import os
import open3d as o3d
import numpy as np
import base64
import tempfile
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from sklearn import neighbors
from pyrfuniverse.envs.base_env import RFUniverseBaseEnv
from common.space_util import image_array_to_point_cloud_intrinsic_matrix, \
    transform_point_cloud
from pyrfuniverse.side_channel import (
    IncomingMessage,
    OutgoingMessage,
)

class ClothDynamicsEnv(RFUniverseBaseEnv):
    def __init__(
            self,
            camera_intrinsic: o3d.camera.PinholeCameraIntrinsic,
            camera_local_to_world_matrix_unity: np.ndarray,
            executable_file='@editor',
            camera_id=221,
            debug=False,
            seed=41234,
            port=5004,
            random_fold_type='tshirt_short',
            grasp_height_range: tuple = (0.5, 1.5),
            grasp_plane_x_offset_range: tuple = (-0.2, 0.2),
            grasp_plane_z_offset_range: tuple = (0.2, 0.5),
            stretch_z_offset: float = 0.3,
            stretch_height: float = 0.9,
            stretch_thr: float = 0.03,
            stretch_max_distance: float = 0.8,
            particle_scale: float = 100,
            fling_end_offset: float = -0.3,
            fling_end_height: float = 0.05,
            fling_speed: float = 0.32,
            fling_min_distance: float = 0.4,
            pick_and_place_height: float = 0.2,
            pick_and_place_speed: float = 0.1,
            medium_gravity: float = 6.0,
            high_gravity: float = 15.0,
            down_speed: float = 0.4,
            time_scale: float = 1.0,
            **kwargs
    ):
        super().__init__(
            executable_file=executable_file,
            port=port
        )
        self.camera_id = camera_id
        self.intrinsic = camera_intrinsic
        np.random.seed(seed)

        # TODO: use this to set camera position and rotation
        # self.step()
        # self.camera = self.GetAttr(id=self.camera_id)
        # postion, rotation = transform_matrix_to_position_and_rotation(camera_local_to_world_matrix_unity)
        # self.camera.SetTransform(postion.tolist())
        # self.camera.SetRotationQuaternion(rotation.as_quat().tolist())

        # TODO: more flexible
        assert random_fold_type in ('tshirt_short', 'tshirt_long')
        self.random_fold_type = random_fold_type

        self.img_width = self.intrinsic.width
        self.img_height = self.intrinsic.height
        self.particle_scale = particle_scale
        self.grasp_height_range = grasp_height_range
        self.grasp_plane_x_offset_range = grasp_plane_x_offset_range
        self.grasp_plane_z_offset_range = grasp_plane_z_offset_range
        self.stretch_z_offset = stretch_z_offset
        self.stretch_height = stretch_height
        self.stretch_thr = stretch_thr
        self.stretch_max_distance = stretch_max_distance
        self.fling_end_offset = fling_end_offset
        self.fling_end_height = fling_end_height
        self.fling_speed = fling_speed
        self.fling_min_distance = fling_min_distance
        self.pick_and_place_height = pick_and_place_height
        self.pick_and_place_speed = pick_and_place_speed
        self.medium_gravity = medium_gravity
        self.high_gravity = high_gravity
        self.down_speed = down_speed
        self.time_scale = time_scale
        self.debug = debug

        self.AddListener('Done', self.receive_done)
        self.AddListener('Depth', self.receive_depth)
        self.AddListener('Particles', self.receive_particles)
        self.AddListener('ParticleRadius', self.receive_particle_radius)
        self.AddListener('InitParticles', self.receive_init_particles)
        self.AddListener('GraspPoint', self.receive_grasp_point)
        self.AddListener('StaticMeshVertices', self.receive_static_vertices)
        self.AddListener('BestGraspPoints', self.receive_best_grasp_points)
        self.SetTimeScale(self.time_scale if not self.debug else 1.0)

        self.step()
        self.extrinsics_unity = self.get_camera_extrinsics_in_unity()
        self.unity_to_open3d_transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.open3d_to_unity_transform = np.linalg.inv(self.unity_to_open3d_transform)

        self.grasp_point_1 = None
        self.grasp_point_2 = None
        self.depth_img = None
        self.particle_pos = None
        self.isDone = False
        self.init_particle_pos = None
        self.category_scale = None
        self.category_center = None
        self.particle_nocs = None
        self.best_grasp_points = None

        self.reset_buffer()

    def reset_buffer(self):
        self.depth_img = None
        self.particle_pos = None
        self.isDone = False

    def set_category_meta(self, meta_dict: dict):
        self.category_scale = meta_dict['category_scale']
        self.category_center = meta_dict['category_center']

    def get_camera_extrinsics_in_unity(self) -> np.ndarray:
        self.step()
        camera = self.GetAttr(id=self.camera_id)
        self.step()
        local_to_world_matrix = camera.data['local_to_world_matrix']
        matrix = np.array(local_to_world_matrix).reshape((4, 4))
        extrinsic = np.linalg.inv(matrix)
        return extrinsic

    def receive_done(self, msg: IncomingMessage):
        self.isDone = True

    def receive_depth(self, msg: IncomingMessage):
        index = msg.read_int32()
        depth_str = msg.read_string()
        depth_bytes = base64.b64decode(depth_str)
        temp_file_path = os.path.join(tempfile.gettempdir(), 'temp_img.exr')
        with open(temp_file_path, 'wb') as f:
            f.write(depth_bytes)
        depth_exr = cv2.imread(temp_file_path, cv2.IMREAD_UNCHANGED)  # (h, w)
        os.remove(temp_file_path)
        self.depth_img = depth_exr

    def receive_static_vertices(self, msg: IncomingMessage):
        vertices = msg.read_float32_list()
        raw_vertices = np.array(vertices).reshape(-1, 3).astype(np.float32)
        # change from Unity coordinate system (left-hand) to Open3D coordinate system (right-hand)
        raw_vertices[:, 0] = -raw_vertices[:, 0]

        assert self.category_center is not None and self.category_scale is not None, 'Category meta not set'
        # normalize vertices to [0, 1] (NOCS space)
        self.particle_nocs = ((raw_vertices - self.category_center) / self.category_scale + \
                              np.array([0.5, 0.5, 0.5], dtype=np.float32)).astype(np.float32)

        # if self.debug:
        #     coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        #     mesh_nocs = o3d.geometry.PointCloud()
        #     mesh_nocs.points = o3d.utility.Vector3dVector(self.particle_nocs)
        #     mesh_nocs.colors = o3d.utility.Vector3dVector(self.particle_nocs)
        #     o3d.visualization.draw_geometries([mesh_nocs, coord])

    def receive_best_grasp_points(self, msg: IncomingMessage):
        best_grasp_points = np.array(msg.read_float32_list()).reshape(-1, 3).astype(np.float32)  # (K, 3)
        is_valid = np.array(msg.read_float32_list()).astype(np.bool)  # (k, )
        valid_grasp_points = best_grasp_points[is_valid, :]  # (N', 3)
        invalid_grasp_points = best_grasp_points[~is_valid, :]  # (N'', 3)
        if valid_grasp_points.shape[0] < 2:
            # randomly select points from invalid_grasp_points and add to valid_grasp_points
            num_to_add = 2 - valid_grasp_points.shape[0]
            if invalid_grasp_points.shape[0] < num_to_add:
                num_to_add = invalid_grasp_points.shape[0]
            idx = np.random.choice(invalid_grasp_points.shape[0], num_to_add, replace=False)
            best_grasp_points = np.concatenate([valid_grasp_points, invalid_grasp_points[idx, :]], axis=0)
        elif valid_grasp_points.shape[0] > 2:
            # randomly select 2 points from valid_grasp_points
            idx = np.random.choice(valid_grasp_points.shape[0], 2, replace=False)
            best_grasp_points = valid_grasp_points[idx, :]
        else:
            best_grasp_points = valid_grasp_points
        assert best_grasp_points.shape[0] == 2, 'Number of grasp points should be 2'
        # change from Unity coordinate system (left-hand) to Open3D coordinate system (right-hand)
        self.best_grasp_points = transform_point_cloud(best_grasp_points, self.unity_to_open3d_transform)

    def receive_particles(self, msg: IncomingMessage):
        particles = msg.read_float32_list()
        particle_pos = np.array(particles).reshape(-1, 3).astype(np.float32)
        # change from Unity coordinate system (left-hand) to Open3D coordinate system (right-hand)
        particle_pos = transform_point_cloud(particle_pos, self.unity_to_open3d_transform)
        self.particle_pos = particle_pos

    def receive_particle_radius(self, msg: IncomingMessage):
        self.particle_radius = msg.read_float32()

    def receive_grasp_point(self, msg: IncomingMessage):
        self.grasp_point_1 = [msg.read_float32(), msg.read_float32(), msg.read_float32()]
        self.grasp_point_2 = [msg.read_float32(), msg.read_float32(), msg.read_float32()]

    def receive_init_particles(self, msg: IncomingMessage):
        particles = msg.read_float32_list()
        particles = np.array(particles).reshape(-1, 3).astype(np.float32)
        # change from Unity coordinate system (left-hand) to Open3D coordinate system (right-hand)
        particles = transform_point_cloud(particles, self.unity_to_open3d_transform)
        self.init_particle_pos = particles

    def load_obj(self, obj_path: str):
        # get intrinsic data
        intrinsic_data = self.intrinsic.intrinsic_matrix.astype(np.float32).reshape(-1).tolist()

        # load mesh with random pose
        self.reset_buffer()
        theta = float(np.random.random() * 360)
        center_point_x = np.random.uniform(*self.grasp_plane_x_offset_range)
        center_point_z = np.random.uniform(*self.grasp_plane_z_offset_range)
        self.SendMessage('LoadMesh', obj_path, self.particle_scale, theta,
                         center_point_x, center_point_z)
        while not self.isDone:
            self.step()

        # make sure that the end point of fling action is aligned with the initial position of cloth particles
        self.SendMessage('GetInitParticles', self.fling_end_offset)
        self.SendMessage('GetDepth', intrinsic_data)  # receive depth in callback
        self.SendMessage('GetParticleRadius')  # receive particle_radius in callback
        self.step()

    def random_grab(self):
        self.reset_buffer()
        self.SendMessage('GetParticles') # receive particle_pos in callback
        self.step()
        assert self.particle_pos is not None, 'No particles received'
        select_idx = np.random.randint(0, len(self.particle_pos))
        # transform particle_pos from Open3D coordinate system (right-hand) to Unity coordinate system (left-hand)
        particle_in_unity = transform_point_cloud(self.particle_pos, self.open3d_to_unity_transform)
        # randomly select a particle
        select_pos = particle_in_unity[select_idx].tolist()
        # randomly select a grasp height
        height = np.random.uniform(*self.grasp_height_range)
        grasp_point_in_air = [np.random.uniform(*self.grasp_plane_x_offset_range),
                              height,
                              np.random.uniform(*self.grasp_plane_z_offset_range)]
        self.reset_buffer()
        self.SendMessage('Grab', select_pos[0], select_pos[1], select_pos[2],
                         grasp_point_in_air[0], grasp_point_in_air[1], grasp_point_in_air[2])
        while not self.isDone:
            self.step()

    def random_fold(self):
        self.reset_buffer()
        # TODO: more flexible
        self.SendMessage('RandomFold', self.random_fold_type == 'tshirt_short')  # randomly fold garment
        while not self.isDone:
            self.step()

    def fling(self, left_position: np.ndarray, right_position: np.ndarray):
        self.reset_buffer()
        left_position = left_position.tolist()
        right_position = right_position.tolist()
        self.SendMessage('SetGravity', 0., -self.medium_gravity, 0.)
        self.step()
        self.SendMessage('Stretch', left_position[0], left_position[1], left_position[2],
                         right_position[0], right_position[1], right_position[2],
                         self.stretch_height, self.stretch_thr, self.stretch_z_offset, self.stretch_max_distance)
        while not self.isDone:
            self.step()

        self.reset_buffer()
        self.SendMessage('GetGraspPosition')
        self.SendMessage('GetParticles')
        self.step()

        cloth_height = np.max(self.particle_pos[:, 1]) - np.min(self.particle_pos[:, 1])
        z_offset = max(float(cloth_height / 2), self.fling_min_distance)
        self.reset_buffer()
        self.SendMessage('DualMove', self.grasp_point_1[0], self.grasp_point_1[1], self.stretch_z_offset-z_offset,
                         self.grasp_point_2[0], self.grasp_point_2[1], self.stretch_z_offset-z_offset,
                         self.fling_speed)  # 0.008
        while not self.isDone:
            self.step()

        self.reset_buffer()
        self.SendMessage('DualMove', self.grasp_point_1[0], self.grasp_point_1[1], self.stretch_z_offset+z_offset,
                         self.grasp_point_2[0], self.grasp_point_2[1], self.stretch_z_offset+z_offset,
                         self.fling_speed)  # 0.008
        while not self.isDone:
            self.step()

        self.reset_buffer()
        self.SendMessage('DualMove', self.grasp_point_1[0], self.fling_end_height, self.fling_end_offset + 0.1,
                         self.grasp_point_2[0], self.fling_end_height, self.fling_end_offset + 0.1,
                         self.down_speed)  # 0.01
        while not self.isDone:
            self.step()

        self.reset_buffer()
        self.SendMessage('DualMove', self.grasp_point_1[0], self.fling_end_height, self.fling_end_offset,
                         self.grasp_point_2[0], self.fling_end_height, self.fling_end_offset, 0.1)  # 0.005
        while not self.isDone:
            self.step()

        self.SendMessage('RemoveAllAttach')
        self.step(50)
        self.SendMessage('SetGravity', 0., -self.high_gravity, 0.)
        self.step(50)

    def get_observation(self) -> dict:
        intrinsic_data = self.intrinsic.intrinsic_matrix.astype(np.float32).reshape(-1).tolist()
        self.reset_buffer()  # clear buffer
        self.SendMessage('GetDepth', intrinsic_data)  # receive depth in callback
        self.SendMessage('GetParticles') # receive particle_pos in callback
        self.step()

        assert self.depth_img is not None
        mask_img = np.zeros_like(self.depth_img).astype(np.uint8)
        # TODO: find a better way to generate mask
        mask_img[self.depth_img > 0] = 1  # generate non-zero mask from depth_img

        fake_rgb_img = np.zeros((self.depth_img.shape[0], self.depth_img.shape[1], 3), dtype=np.uint8)
        foreground_pcd = image_array_to_point_cloud_intrinsic_matrix(fake_rgb_img,
                                                                     self.depth_img,
                                                                     self.intrinsic,
                                                                     self.extrinsics_unity)
        # convert from Unity coordinate system to Open3D coordinate system
        foreground_pcd.transform(self.unity_to_open3d_transform)
        pc_xyz = np.asarray(foreground_pcd.points).astype(np.float32)
        # calculate pc_nocs
        pc_nocs = self.calc_pc_nocs(pc_xyz, self.particle_pos, self.particle_nocs)
        if self.debug:
            # heat = cv2.normalize(self.depth_img.copy(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # depth_vis = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            # cv2.imshow('depth', depth_vis)
            # cv2.waitKey()

            coorninate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=(0., 0., 0.))

            particle_pcd = o3d.geometry.PointCloud()
            particle_pcd.points = o3d.utility.Vector3dVector(self.particle_pos)
            particle_pcd.colors = o3d.utility.Vector3dVector(self.particle_nocs)

            foreground_pcd.colors = o3d.utility.Vector3dVector(pc_nocs)
            # o3d.visualization.draw_geometries([foreground_pcd, coorninate])

            if self.best_grasp_points is not None:
                # visualize two spheres from self.best_grasp_points
                sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                sphere1.translate(self.best_grasp_points[0])
                sphere1.paint_uniform_color([1, 0, 0])
                sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                sphere2.translate(self.best_grasp_points[1])
                sphere2.paint_uniform_color([0, 1, 0])
                o3d.visualization.draw_geometries([particle_pcd, coorninate, sphere1, sphere2])
            else:
                o3d.visualization.draw_geometries([coorninate, particle_pcd])


        out_dict = dict(mask_img=mask_img.astype(np.float32),
                        pc_xyz=pc_xyz,
                        pc_nocs=pc_nocs,
                        particle_xyz=self.particle_pos.astype(np.float32),
                        particle_nocs=self.particle_nocs.astype(np.float32))
        return out_dict

    @staticmethod
    def calc_pc_nocs(pc_xyz: np.ndarray,
                     particle_xyz: np.ndarray,
                     particle_nocs: np.ndarray) -> np.ndarray:
        """Find NOCS coordinate for pc (point cloud), each point in pc_xyz will find the closed point in particle_xyz,
            then the corresponding NOCS coordinate is particle_nocs
        """
        knn = neighbors.KNeighborsRegressor(1, weights='distance')
        pc_nocs = knn.fit(particle_xyz, particle_nocs).predict(pc_xyz).astype(np.float32)
        return pc_nocs

