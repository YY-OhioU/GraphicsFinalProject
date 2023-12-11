import time
import warnings

import numpy as np
import pygame as pg
import taichi as ti
import torch
from einops import rearrange
from scipy.spatial.transform import Rotation as R

from modules.rendering import render
from modules.utils import depth2img
from utils.ray_utils import get_ray_directions, get_rays

warnings.filterwarnings("ignore")

IMG_WH = (180, 320)

MANUAL_K = np.array([[275.6894, 0.0000, 91.4382],
                     [0.0000, 273.1759, 160.8402],
                     [0.0000, 0.0000, 1.0000]])

MANUAL_POSE = np.array([[0.91064833, -0.15844536, 0.38159499, -0.66021108],
                        [0.33583007, 0.82185739, -0.46018319, 0.94010761],
                        [-0.24070275, 0.5472161, 0.80163374, -1.60078464],
                        [0., 0., 0., 1.]])

MANUAL_R = 2.0
MANUAL_CENTER = np.array([-0.09299416, -0.03178227, 0.01849253])


@ti.kernel
def write_buffer(W: ti.i32, H: ti.i32, x: ti.types.ndarray(),
                 final_pixel: ti.template()):
    for i, j in ti.ndrange(W, H):
        for p in ti.static(range(3)):
            final_pixel[i, j][p] = x[H - j, i, p]


class OrbitCamera:

    def __init__(self, K, img_wh, r):
        self.K = K
        self.W, self.H = img_wh
        self.radius = r
        self.center = MANUAL_CENTER

        # pose_np = poses.cpu().numpy()
        pose_np = MANUAL_POSE
        # choose a pose as the initial rotation
        self.rot = pose_np[:3, :3]

        self.rotate_speed = 0.005
        self.res_defalut = pose_np

    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4)
        rot[:3, :3] = self.rot
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    def reset(self, pose=None):
        self.rot = np.eye(3)
        self.center = np.zeros(3)
        self.radius = 2.0
        if pose is not None:
            self.rot = pose.cpu().numpy()[:3, :3]

    def orbit(self, dx, dy):
        rotvec_x = self.rot[:, 1] * np.radians(100 * self.rotate_speed * dx)
        rotvec_y = self.rot[:, 0] * np.radians(-100 * self.rotate_speed * dy)

        self.rot = R.from_rotvec(rotvec_y).as_matrix() @ \
                   R.from_rotvec(rotvec_x).as_matrix() @ \
                   self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 1e-4 * self.rot @ np.array([dx, dy, dz])


class NGPGUI:

    def __init__(
            self,
            model,
            K=MANUAL_K,
            img_wh=IMG_WH,
            radius=MANUAL_R
    ):
        self.model = model

        # print(f"loading ckpt from: {hparams.ckpt_path}")
        # state_dict = torch.load(hparams.ckpt_path)
        # self.model.load_state_dict(state_dict)

        self.cam = OrbitCamera(K, img_wh, r=radius)
        self.W, self.H = img_wh
        self.render_buffer = ti.Vector.field(
            n=3,
            dtype=float,
            shape=(self.W, self.H)
        )

        self.exp_step_factor = 1 / 256
        # if self.hparams.dataset_name in ['colmap', 'nerfpp']:
        #     self.exp_step_factor = 1 / 256
        # else:
        #     self.exp_step_factor = 0

        # placeholders
        self.dt = 0
        self.mean_samples = 0
        self.img_mode = 0

        self.rendering = True

    def render_cam(self):
        t = time.time()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            directions = get_ray_directions(
                self.cam.H,
                self.cam.W,
                self.cam.K,
                device='cuda'
            )
            rays_o, rays_d = get_rays(
                directions,
                torch.cuda.FloatTensor(self.cam.pose)
            )
            results = render(
                self.model,
                rays_o,
                rays_d,
                test_time=True,
                exp_step_factor=self.exp_step_factor,
            )

        # torch.cuda.synchronize()
        self.dt = time.time() - t
        self.mean_samples = results['total_samples'] / len(rays_o)

        if self.img_mode == 0:
            rgb = rearrange(results["rgb"], "(h w) c -> h w c", h=self.H)
            return rgb.cpu().numpy()
            # return rgb
        assert self.img_mode == 1
        depth = rearrange(results["depth"], "(h w) -> h w", h=self.H)
        # return depth.cpu().numpy().astype(np.float32)
        return depth2img(depth.cpu().numpy()).astype(np.float32)

    def check_cam_rotate(self, window, last_orbit_x, last_orbit_y):
        if window.is_pressed(ti.ui.RMB):
            curr_mouse_x, curr_mouse_y = window.get_cursor_pos()
            if last_orbit_x is None or last_orbit_y is None:
                last_orbit_x, last_orbit_y = curr_mouse_x, curr_mouse_y
            else:
                dx = curr_mouse_x - last_orbit_x
                dy = curr_mouse_y - last_orbit_y
                self.cam.orbit(dx, -dy)
                last_orbit_x, last_orbit_y = curr_mouse_x, curr_mouse_y
        else:
            last_orbit_x = None
            last_orbit_y = None

        return last_orbit_x, last_orbit_y

    def check_key_press(self, pressed_key):
        if pressed_key[pg.K_w]:
            self.cam.scale(0.2)
        if pressed_key[pg.K_s]:
            self.cam.scale(-0.2)
        if pressed_key[pg.K_a]:
            self.cam.pan(100, 0.)
        if pressed_key[pg.K_d]:
            self.cam.pan(-100, 0.)
        if pressed_key[pg.K_e]:
            self.cam.pan(0., -100)
        if pressed_key[pg.K_q]:
            self.cam.pan(0., 100)
        if pressed_key[pg.K_c]:
            print(self.cam.pose)
            print(self.cam.radius)
            print(self.cam.center)

    # def render(self):
    #
    #     window = ti.ui.Window(
    #         'taichi_ngp',
    #         (self.W, self.H),
    #     )
    #     canvas = window.get_canvas()
    #     gui = window.get_gui()
    #
    #     # GUI controls variables
    #     last_orbit_x = None
    #     last_orbit_y = None
    #
    #     view_id = 0
    #     last_view_id = 0
    #
    #     views_size = self.poses.shape[0] - 1
    #
    #     while window.running:
    #         self.check_key_press(window)
    #         last_orbit_x, last_orbit_y = self.check_cam_rotate(
    #             window, last_orbit_x, last_orbit_y)
    #
    #         # use img_mode to switch between depth
    #
    #         with gui.sub_window("Control", 0.01, 0.01, 0.4, 0.2) as w:
    #             self.cam.rotate_speed = w.slider_float('rotate speed',
    #                                                    self.cam.rotate_speed,
    #                                                    0.1, 1.)
    #
    #             self.img_mode = w.checkbox("show depth", self.img_mode)
    #
    #             view_id = w.slider_int('train view', view_id, 0, views_size)
    #
    #             if last_view_id != view_id:
    #                 last_view_id = view_id
    #                 self.cam.reset(self.poses[view_id])
    #
    #             w.text(f'samples per rays: {self.mean_samples:.2f} s/r')
    #             w.text(f'render times: {1000*self.dt:.2f} ms')
    #
    #         ngp_buffer = self.render_cam()
    #         write_buffer(self.W, self.H, ngp_buffer, self.render_buffer)
    #         canvas.set_image(self.render_buffer)
    #         window.show()

# if __name__ == "__main__":
#     ti.init(arch=ti.cuda, device_memory_GB=4)
#
#     hparams = get_opts()
#     dataset = dataset_dict[hparams.dataset_name](
#         root_dir=hparams.root_dir,
#         downsample=hparams.downsample,
#         read_meta=True,
#     )
#
#     NGPGUI(hparams, dataset.K, dataset.img_wh, dataset.poses).render()
