"""
author: JDachuk
created: 7/19/19
"""

import random
from pyglet.sprite import Sprite
from support import *


class PhysicObject(Sprite):
    def __init__(self, img, velocity_x=0., velocity_y=0., *args, **kwargs):
        super().__init__(img=img, *args, **kwargs)
        self.velocity_x, self.velocity_y = velocity_x, velocity_y

    def animate(self, dt):
        self._animate(dt)

    @property
    def bbox(self):
        return self.x, self.y, self.x + self.width, self.y + self.height


class TubeHead(PhysicObject):
    def __init__(self, *args, **kwargs):
        super().__init__(img=Images.TUBE_END_IMG, velocity_x=-Const.FRAMES_PER_SECOND, *args, **kwargs)

    def _animate(self, dt):
        self.x += self.velocity_x * dt


class TubeBody(PhysicObject):
    def __init__(self, *args, **kwargs):
        super().__init__(img=Images.TUBE_IMG, velocity_x=-Const.FRAMES_PER_SECOND, *args, **kwargs)

    def _animate(self, dt):
        self.x += self.velocity_x * dt


class Tube:
    def __init__(self):
        self.height = random.randint(46, Const.HEIGHT - Const.GATE_SIZE - 46)
        self.bottom_head = TubeHead(x=Const.WIDTH, y=self.height)
        self.bottom_body = TubeBody(x=Const.WIDTH + 3, y=0)
        self.bottom_body.scale_y = (self.height + self.bottom_head.height) / self.bottom_body.height
        self.top_head = TubeHead(x=Const.WIDTH, y=self.height + Const.GATE_SIZE)
        self.top_body = TubeBody(x=Const.WIDTH + 3, y=self.height + Const.GATE_SIZE)
        self.top_body.scale_y = (Const.HEIGHT - self.top_head.height) / self.top_body.height
        self.center = (self.top_head.x + self.top_head.width) / 2
        self.crossed = False
        self.half_crossed = False

    def draw(self):
        self.bottom_body.draw()
        self.bottom_head.draw()

        self.top_body.draw()
        self.top_head.draw()

    def animate(self, dt):
        self.bottom_body.animate(dt)
        self.bottom_head.animate(dt)

        self.top_body.animate(dt)
        self.top_head.animate(dt)

        self.center = (self.top_head.x + self.top_head.width) / 2

    @property
    def bbox(self):
        return [self.top_body.bbox, self.top_head.bbox, self.bottom_head.bbox, self.bottom_body.bbox]

    @property
    def x(self):
        return self.bottom_head.x


class Land(Sprite):
    def __init__(self, *args, **kwargs):
        img_seq = pyglet.image.ImageGrid(Images.LAND_IMG, 1, 13, 450, 55, 0, 1)
        anim = pyglet.image.Animation.from_image_sequence(img_seq, Const.FRAME_RATE)
        super().__init__(img=anim, y=-10, *args, **kwargs)

