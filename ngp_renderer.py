import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import taichi as ti
import torch

from gui import NGPGUI
from modules.networks import NGP

CWD = Path(__file__).parent


def taichi_init(*args, **kwargs):
    taichi_init_args = {"arch": ti.cuda}
    ti.init(**taichi_init_args)


def get_ngp_renderer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set seed
    seed = 23
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    taichi_init()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model_config = {
        'scale': 16.0,
        'pos_encoder_type': 'hash',
        'max_res': 4096,
        'half_opt': False,
    }
    model = NGP(**model_config).to(device)
    weights = CWD / "models" / "model.pth"
    state_dict = torch.load(weights)
    model.load_state_dict(state_dict)

    ti.reset()
    taichi_init()

    ngp_gui = NGPGUI(
        model,
    )
    return ngp_gui


if __name__ == '__main__':
    ns = SimpleNamespace(
        root_dir='data',
        dataset_name='colmap',
        split='train',
        downsample=0.25,
        model_namp='ngp',
        scale=16.0, half_opt=False, encoder_type='hash', sh_degree=2, grid_size=256, grid_radius=0.0125, origin_sh=0.0,
        origin_sigma=0.1, distortion_loss_w=0, batch_size=4096, ray_sampling_strategy='all_images', max_steps=3000,
        lr=0.01, random_bg=False, exp_name='custom', gpu=0, ckpt_path='..\\NeRF-OBJ-Viewer\\models\\model.pth',
        gui=True, deployment=False, deployment_model_path='./'

    )
    npg_renderer = get_ngp_renderer(ns)
    npg_renderer = get_ngp_renderer()
    npg_renderer.render()
