import numpy as np
import pygame as pg

from ngp_renderer import get_ngp_renderer


class NGPScene:
    def __init__(self, app):
        self.app = app
        self.ctx = self.app.ctx

        self.ngp_render = get_ngp_renderer()
        self.W = self.ngp_render.W
        self.H = self.ngp_render.H

        self.frame_texture = self.ctx.texture((self.W, self.H), 3, dtype='f4')
        self.depth_mode = 0
        self.cam_rotating = False
        self.mouse_last_x = None
        self.mouse_last_y = None

        with open('shaders/quad.vert') as f:
            vertex = f.read()
        with open('shaders/quad.frag') as f:
            fragment = f.read()
        self.program = self.ctx.program(vertex_shader=vertex, fragment_shader=fragment)

        self.vbo = self.ctx.buffer(np.array([
            # x    y     u  v
            -1.0, -1.0, 0, 0,  # lower left
            1.0, -1.0, 1, 0,  # lower right
            1.0, 1.0, 1, 1,  # upper right
            -1.0, 1.0, 0, 1,  # upper left
            -1.0, -1.0, 0, 0,  # lower left
            1.0, 1.0, 1, 1,  # upper right

        ], dtype="f4"))
        self.vao = self.ctx.vertex_array(self.program, [self.vbo.bind('in_vert', 'in_texcoord')])

    def update(self):
        if self.cam_rotating:
            curr_mouse_x, curr_mouse_y = pg.mouse.get_pos()
            if self.mouse_last_x is None or self.mouse_last_y is None:
                self.mouse_last_x, self.mouse_last_y = curr_mouse_x, curr_mouse_y
            else:
                dx = curr_mouse_x - self.mouse_last_x
                dy = curr_mouse_y - self.mouse_last_y
                self.ngp_render.cam.orbit(dx, dy)
                self.mouse_last_x, self.mouse_last_y = curr_mouse_x, curr_mouse_y
        else:
            self.mouse_last_x = None
            self.mouse_last_x = None

    def render(self):
        self.ctx.clear(color=(0.08, 0.16, 0.18))
        # rgb = np.random.rand(self.H, self.W, 3).astype('f4')
        # rgb = self.ngp_render.render_cam().cpu().numpy()
        rgb = self.ngp_render.render_cam()

        self.frame_texture.write(np.flipud(rgb).tobytes())
        self.frame_texture.use(location=0)

        self.vao.render()
        pg.display.flip()

    def destroy(self):
        self.vbo.release()
        self.program.release()
        self.vao.release()

    def handle_key_pressed(self, pressed_key):
        self.ngp_render.check_key_press(pressed_key)

    def on_event(self, event):
        if event.type == pg.KEYDOWN and event.key == pg.K_b:
            self.flip_depth_mode()

        if event.type == pg.MOUSEBUTTONDOWN and event.button == 3:  # RMB is pressed
            self.cam_rotating = True

        if event.type == pg.MOUSEBUTTONUP and event.button == 3:  # RMB is released
            self.cam_rotating = False

    def flip_depth_mode(self):
        self.depth_mode = (self.depth_mode + 1) % 2
        self.ngp_render.img_mode = self.depth_mode
