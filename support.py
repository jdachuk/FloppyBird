"""
author: JDachuk
created: 7/19/19
"""

import pyglet

pyglet.resource.path = ['resources']
pyglet.resource.reindex()


class Const:
    VERSION = 1.0

    WIDTH = 450
    HEIGHT = 600
    BIRD_X = 100
    BIRD_Y = 200
    G_A = 9.80954
    FRAMES_PER_SECOND = 120
    FRAME_RATE = 1 / FRAMES_PER_SECOND

    TUBE_PERIOD = 4.3
    GATE_SIZE = 180

    POPULATION_SIZE = 50
    MUTATION_RATE = .01
    MU, SIGMA = 0, 1


class Images:
    BIRD_IMG = pyglet.resource.image('bird.png')
    TUBE_END_IMG = pyglet.resource.image('tube_end.png')
    TUBE_IMG = pyglet.resource.image('tube.png')
    BACKGROUND_IMG = pyglet.resource.image('background.png')
    LAND_IMG = pyglet.resource.image('ground.png')
