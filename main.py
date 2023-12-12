import sys

import moderngl as mgl
import pygame as pg

from ngp_scene import NGPScene
from obj_scene import ObjScene


class App:
    def __init__(self, win_size=(360, 640)):
        # opengl context
        self.win_size = win_size
        pg.display.set_mode(win_size, flags=pg.OPENGL | pg.DOUBLEBUF)
        self.ctx = mgl.create_context()

        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        # create opengl context
        pg.display.set_mode(self.win_size, flags=pg.OPENGL | pg.DOUBLEBUF)
        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)

        # time objects
        self.clock = pg.time.Clock()

        self.ngp_scene = NGPScene(self)
        self.obj_scene = ObjScene(self)
        self.scene_list = [self.ngp_scene, self.obj_scene]

        self.current_scene = self.ngp_scene

        self.scene_id = 0
        self.delta_time = 0

    def switch_scene(self):
        self.current_scene = self.scene_list[self.scene_id]
        self.scene_id = (self.scene_id + 1) % 2

    def render(self):
        self.current_scene.render()

    def update(self):
        self.current_scene.update()

    def run(self):
        # self.ngp_scene.flip_depth_mode()
        while True:
            self.ctx.clear(color=(0.08, 0.16, 0.18))
            self.check_events()
            self.update()
            self.render()
            # self.clock.tick(0)
            fps = self.clock.get_fps()
            pg.display.set_caption(f'{fps :.1f}')
            self.delta_time = self.clock.tick(0)

    def destroy(self):
        self.current_scene.destroy()

    def on_exit(self):
        self.destroy()
        pg.quit()
        sys.exit()

    def check_events(self):
        pressed_keys = pg.key.get_pressed()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.on_exit()
            self.current_scene.on_event(event)
            if event.type == pg.KEYDOWN and event.key == pg.K_r:
                self.switch_scene()

        if pressed_keys[pg.K_ESCAPE]:
            self.on_exit()

        self.current_scene.handle_key_pressed(pressed_keys)


if __name__ == '__main__':
    app = App()
    app.run()
