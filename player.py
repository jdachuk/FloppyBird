"""
author: edacjos
created: 7/19/19
"""

import time
import random
import numpy as np
from support import *
from game_objects import PhysicObject


class BirdBrain:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_der(x):
        return BirdBrain.sigmoid(x) * (1 - BirdBrain.sigmoid(x))

    def __init__(self):
        self.i_w = 2 * np.random.random((6, 3)) - 1
        self.h_w = 2 * np.random.random((4, 2)) - 1
        self.o_w = 2 * np.random.random((3, 2)) - 1

    def analyze(self, input_data):
        input_data = np.append(input_data, np.ones((1, 1)))

        input_activation = np.dot(input_data, self.i_w)
        input_activation = self.sigmoid(input_activation)
        input_activation = np.append(input_activation, np.ones((1, 1)))

        hidden_activation = np.dot(input_activation, self.h_w)
        hidden_activation = self.sigmoid(hidden_activation)
        hidden_activation = np.append(hidden_activation, np.ones((1, 1)))

        output_activation = np.dot(hidden_activation, self.o_w)
        output_activation = self.sigmoid(output_activation)

        return output_activation

    def clone(self):
        clone = BirdBrain()
        clone.i_w = self.i_w
        clone.h_w = self.h_w
        clone.o_w = self.o_w
        return clone

    def mutate(self):
        def weights_mutate(weights):
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    if random.random() < Const.MUTATION_RATE:
                        mutation_factor = np.random.normal(Const.MU, Const.SIGMA)

                        weights[i][j] += mutation_factor
                        if weights[i][j] > 1:
                            weights[i][j] = 1
                        elif weights[i][j] < -1:
                            weights[i][j] = -1
            return weights

        self.i_w = weights_mutate(self.i_w)
        self.h_w = weights_mutate(self.h_w)
        self.o_w = weights_mutate(self.o_w)

    def crossover(self, other):

        def weights_crossover(weights, other_w):
            child_w = np.random.random(weights.shape)

            rand_row = random.randint(0, child_w.shape[0])
            rand_col = random.randint(0, child_w.shape[1])

            for i in range(child_w.shape[0]):
                for j in range(child_w.shape[1]):
                    if i < rand_row or (i == rand_row and j <= rand_col):
                        child_w[i][j] = weights[i][j]
                    else:
                        child_w[i][j] = other_w[i][j]

            return child_w

        child = BirdBrain()

        child.i_w = weights_crossover(self.i_w, other.i_w)
        child.h_w = weights_crossover(self.h_w, other.h_w)
        child.o_w = weights_crossover(self.o_w, other.o_w)

        return child

    def save_to_dict(self):
        def to_py_arr(ndarray):
            array = []
            for i in range(ndarray.shape[0]):
                row = []
                for j in range(ndarray.shape[1]):
                    row.append(ndarray[i][j])
                array.append(row)
            return array

        result = {
            'version': Const.VERSION,
            'weights_input_shape': self.i_w.shape,
            'weights_hidden_shape': self.h_w.shape,
            'weights_output_shape': self.o_w.shape,
            'weights_input': to_py_arr(self.i_w),
            'weights_hidden': to_py_arr(self.h_w),
            'weights_output': to_py_arr(self.o_w)
        }
        return result

    def load_from_dict(self, dictionary):
        if int(dictionary['version']) != int(Const.VERSION):
            raise ValueError('Inconsistent versions!')
        self.i_w = np.array(dictionary['weights_input'])
        self.h_w = np.array(dictionary['weights_hidden'])
        self.o_w = np.array(dictionary['weights_output'])


class Bird(PhysicObject):
    def __init__(self, game, *args, **kwargs):
        super().__init__(img=Images.BIRD_IMG, *args, **kwargs)
        self.game = game
        self.x, self.y = 100, 400
        self.fly_acceleration = 4
        self.alive = True
        self.brain = BirdBrain()
        self.birth_time = 0
        self.fitness = 0
        self.birth_time = time.time()

    def _animate(self, dt):
        self.velocity_y -= Const.G_A * dt
        self.y += self.velocity_y

    def animate(self, dt):
        if self.alive:
            data = self.collect_data()
            self.make_decision(data)
        self._animate(dt)

    def accelerate(self):
        if not self.alive:
            return
        self.velocity_y = self.fly_acceleration

    def make_decision(self, input_data):
        brain_output = self.brain.analyze(input_data)

        if brain_output[0] > brain_output[1]:
            self.accelerate()
        else:
            pass

    def collect_data(self):
        closest_tube = self.game.get_closest_tube()
        tube_bbox = closest_tube.bbox
        bird_bbox = self.bbox

        return np.array([
            tube_bbox[1][0] - bird_bbox[2],  # distance to nearest tube
            tube_bbox[1][2] - bird_bbox[0],  # distance to the end of nearest tube
            tube_bbox[1][1] - bird_bbox[3],  # distance to top of nearest gates
            tube_bbox[2][3] - bird_bbox[1],  # distance to bottom of nearest gates
            self.velocity_y                  # self falling velocity
        ])

    def clone(self):
        clone = Bird(self.game)
        clone.brain = self.brain.clone()
        return clone

    def crossover(self, other):
        child = Bird(self.game)
        child.brain = self.brain.crossover(other.brain)
        return child

    def mutate(self):
        self.brain.mutate()

    def check_collisions(self):
        bird_x1, bird_y1, bird_x2, bird_y2 = self.bbox

        if bird_y1 <= 55:
            self.die()

        tube = self.game.get_closest_tube()
        for x1, y1, x2, y2 in tube.bbox:
            if x1 <= bird_x1 <= x2 or x1 <= bird_x2 <= x2:
                if y1 <= bird_y1 <= y2 or y1 <= bird_y2 <= y2 or bird_y1 > Const.HEIGHT:
                    self.die()

    def die(self):
        if self.alive:
            self.alive = False
            self.velocity_y = -self.velocity_y / 2
            self.fitness = time.time() - self.birth_time
