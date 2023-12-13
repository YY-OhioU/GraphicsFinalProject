import pickle
from pathlib import Path

import glm
import moderngl as mgl
import numpy as np
import pygame as pg
import pywavefront

from light import Light
from obj_cam import Camera

CWD = Path(__file__).parent
FOV = 90  # deg
NEAR = 0.1
FAR = 200
SPEED = 0.005
SENSITIVITY = 0.04


# plydata = PlyData.read(CWD / "models" / "School_point_cloud.ply")


class BaseModel:
    vao = None
    vbo = None
    texture = None
    shader_name = 'default'

    def __init__(self, scene, pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
        self.scene = scene
        self.app = scene.app
        # self.app = scene
        self.ctx = self.app.ctx
        self.pos = pos
        self.rot = glm.vec3([glm.radians(a) for a in rot])
        self.scale = scale
        self.m_model = self.get_model_matrix()
        self.program = self.get_shader()
        self.camera = scene.cam

        self.load_vao()
        self.load_texture()

    def get_shader(self):
        shader_folder = CWD / 'shaders'
        name = self.shader_name
        with open(shader_folder / f'{name}.vert') as file:
            vertex_shader = file.read()
        with open(shader_folder / f'{name}.frag') as file:
            fragment_shader = file.read()
        program = self.app.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        return program

    def load_vbo(self):
        raise NotImplemented

    def load_texture(self):
        raise NotImplemented

    def load_vao(self, force=False):
        if self.vao and not force:
            return
        vertex_data, d_format, attribs = self.load_vbo()
        self.vbo = self.app.ctx.buffer(vertex_data)
        self.vao = self.app.ctx.vertex_array(self.program, [(self.vbo, d_format, *attribs)], skip_errors=True)

    def on_init(self):
        self.program['m_view_light'].write(self.scene.light.m_view_light)
        # resolution
        self.program['u_resolution'].write(glm.vec2(self.app.win_size))

        # texture
        self.program['u_texture_0'] = 0
        self.texture.use(location=0)
        # mvp
        self.program['m_proj'].write(self.camera.m_proj)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)
        # light
        self.program['light.position'].write(self.scene.light.position)
        self.program['light.Ia'].write(self.scene.light.Ia)
        self.program['light.Id'].write(self.scene.light.Id)
        self.program['light.Is'].write(self.scene.light.Is)

    def update(self, *args, **kwargs):
        self.texture.use(location=0)
        self.program['camPos'].write(self.camera.position)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)

    def get_model_matrix(self):
        m_model = glm.mat4()
        # translate
        m_model = glm.translate(m_model, self.pos)
        # rotate
        m_model = glm.rotate(m_model, self.rot.z, glm.vec3(0, 0, 1))
        m_model = glm.rotate(m_model, self.rot.y, glm.vec3(0, 1, 0))
        m_model = glm.rotate(m_model, self.rot.x, glm.vec3(1, 0, 0))
        # scale
        m_model = glm.scale(m_model, self.scale)
        return m_model

    def render(self, *args):
        self.update(self.camera)
        self.vao.render()


class Cat(BaseModel):
    shader_name = 'default'

    def __init__(self, app, pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
        super().__init__(app, pos, rot, scale)
        self.on_init()

    def load_vbo(self):
        obj_file = CWD / 'models' / 'cat' / '20430_Cat_v1_NEW.obj'
        objs = pywavefront.Wavefront(obj_file, cache=True, parse=True)
        obj = objs.materials.popitem()[1]
        vertex_data = np.array(obj.vertices, dtype='f4')
        d_format = '2f 3f 3f'
        attribs = ['in_texcoord_0', 'in_normal', 'in_position']
        return vertex_data, d_format, attribs

    def load_texture(self):
        texture_file = CWD / 'models' / 'cat' / '20430_cat_diff_v1.jpg'
        texture = pg.image.load(texture_file).convert()
        texture = pg.transform.flip(texture, flip_x=False, flip_y=True)
        texture = self.ctx.texture(size=texture.get_size(), components=3,
                                   data=pg.image.tostring(texture, 'RGB'))
        texture.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
        texture.build_mipmaps()
        texture.anisotropy = 32.0

        self.texture = texture

    def update(self, *args, **kwargs):
        self.m_model = self.get_model_matrix()
        super().update()


class Campus(BaseModel):
    shader_name = 'point_cloud'

    def __init__(self, app, pos=(0, 0, 0), rot=(0, 0, 0), scale=(0.01, 0.01, 0.01)):
        super().__init__(app, pos, rot, scale)
        self.on_init()

    def load_texture(self):
        self.texture = None

    def load_vbo(self):
        # plydata = PlyData.read(CWD / "models" / "School_point_cloud.ply")
        # vertex_data = np.array(plydata['vertex']).astype(
        #     [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')])
        obj_file = CWD / 'models' / 'point_cloud.pkl'
        with open(obj_file, 'rb') as f:
            vertex_data = pickle.load(f)
        d_format = '3f8 3f'
        attribs = ['in_position', 'in_color']
        return vertex_data, d_format, attribs

    def on_init(self):
        # mvp
        self.program['m_proj'].write(self.camera.m_proj)
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)

    def update(self, *args, **kwargs):
        self.program['m_view'].write(self.camera.m_view)
        self.program['m_model'].write(self.m_model)

    def render(self, *args):
        self.update(self.camera)
        self.vao.render(mgl.POINTS)


class ObjScene:
    def __init__(self, app):
        self.campus = None
        self.cat = None
        self.app = app
        self.ctx = self.app.ctx
        self.objs = []
        self.light = Light()

        self.load_scene()
        self.time = 0

    def get_time(self):
        self.time = pg.time.get_ticks() * 0.001

    def load_scene(self):
        self.cam = Camera(self.app)

        # Cat(app, pos=(0, -1, -10), rot=(-90, 0, 0))
        self.cat = Cat(self, pos=(0, -1.5, -20), rot=(-90, 0, 0), scale=(0.3, 0.3, 0.3))
        self.campus = Campus(self, pos=(0, -1, -10), rot=(0, 150, 0), scale=(1, 1, 1))
        self.objs.append(self.cat)
        self.objs.append(self.campus)

    def render(self):
        self.app.ctx.screen.use()
        for wo in self.objs:
            wo.render()

        pg.display.flip()

    def update(self):
        self.get_time()
        self.cam.update()
        self.cat.rot.y = self.time

    def destroy(self):
        for obj in self.objs:
            obj.vbo.release()
            obj.vao.release()
            obj.program.release()

    def on_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 3:  # RMB is pressed
            self.cam.rotating = True

        if event.type == pg.MOUSEBUTTONUP and event.button == 3:  # RMB is released
            self.cam.rotating = False
            self.cam.mouse_last_x = None
            self.cam.mouse_last_y = None

    def handle_key_pressed(self, pressed_keys):
        self.cam.move(pressed_keys)
