"""
author: edacjos
created: 7/15/19
"""

import pyglet
import random
import numpy as np
import time
from pyglet.sprite import Sprite
from pyglet.window import key, Window
from pyglet.text import Label

pyglet.resource.path = ['resources']
pyglet.resource.reindex()


class Const:
    WIDTH = 450
    HEIGHT = 600
    BIRD_X = 100
    BIRD_Y = 200
    G_A = 9.80954
    FRAMES_PER_SECOND = 60
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


class PhysicObject(Sprite):
    def __init__(self, img, velocity_x=0., velocity_y=0., *args, **kwargs):
        super().__init__(img=img, *args, **kwargs)
        self.velocity_x, self.velocity_y = velocity_x, velocity_y

    def animate(self, dt):
        self._animate(dt)

    @property
    def bbox(self):
        return self.x, self.y, self.x + self.width, self.y + self.height


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


class Bird(PhysicObject):
    def __init__(self, *args, **kwargs):
        super().__init__(img=Images.BIRD_IMG, *args, **kwargs)
        self.x, self.y = 100, 400
        self.fly_acceleration = 4
        self.alive = True

    def _animate(self, dt):
        self.velocity_y -= Const.G_A * dt
        self.y += self.velocity_y

    def accelerate(self):
        if not self.alive:
            return
        self.velocity_y = self.fly_acceleration

    def die(self):
        if self.alive:
            self.alive = False
            self.velocity_y = -self.velocity_y / 2


class CoolBird(Bird):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.brain = BirdBrain()
        self.birth_time = 0
        self.fitness = 0

    def draw(self):
        self.birth_time = time.time()
        super().draw()

    def animate(self, dt):
        if self.alive:
            data = self.collect_data()
            self.make_decision(data)
        self._animate(dt)

    def make_decision(self, input_data):
        brain_output = self.brain.analyze(input_data)

        if brain_output[0] > brain_output[1]:
            self.accelerate()
        else:
            pass

    def collect_data(self):
        closest_tube = window.get_closest_tube()
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
        clone = CoolBird()
        clone.brain = self.brain.clone()
        return clone

    def crossover(self, other):
        child = CoolBird()
        child.brain = self.brain.crossover(other.brain)
        return child

    def mutate(self):
        self.brain.mutate()

    def die(self):
        if self.alive:
            self.alive = False
            self.velocity_y = -self.velocity_y / 2
            self.fitness = time.time() - self.birth_time

    def check_collisions(self):
        bird_x1, bird_y1, bird_x2, bird_y2 = self.bbox

        if bird_y1 <= 55 or bird_y1 >= Const.HEIGHT:
            self.die()

        tube = window.get_closest_tube()
        for x1, y1, x2, y2 in tube.bbox:
            if x1 <= bird_x1 <= x2 or x1 <= bird_x2 <= x2:
                if y1 <= bird_y1 <= y2 or y1 <= bird_y2 <= y2:
                    self.die()


class Population:
    def __init__(self):
        self.birds = []
        self.top_bird = None
        self.size = Const.POPULATION_SIZE
        self.generation_id = 0
        self.total_fitness = 0

        self.create_birds()

    def create_birds(self):
        for _ in range(self.size):
            self.birds.append(CoolBird())

    def draw(self):
        for bird in self.birds:
            bird.draw()

    def animate(self, dt):
        for bird in self.birds:
            bird.animate(dt)

    def check_collisions(self):
        for bird in self.birds:
            bird.check_collisions()

        done = True
        for bird in self.birds:
            done = done and not bird.alive

        if done:
            self.natural_selection()

    def update_fitness(self):
        self.total_fitness = 0

        for bird in self.birds:
            self.total_fitness += bird.fitness

    def select_top_bird(self):
        max_fitness, max_idx = 0, 0
        for idx, bird in enumerate(self.birds):
            if bird.fitness > max_fitness:
                max_fitness = bird.fitness
                max_idx = idx

        self.top_bird = self.birds[max_idx]

    def select_bird(self):
        rand = self.total_fitness * random.random()

        cum_sum = 0
        for bird in self.birds:
            cum_sum += bird.fitness
            if cum_sum >= rand:
                return bird

    def natural_selection(self):
        self.update_fitness()
        self.select_top_bird()

        new_generation = [self.top_bird.clone()]

        while len(new_generation) < self.size:
            parent1 = self.select_bird()
            parent2 = self.select_bird()

            child = parent1.crossover(parent2)

            child.mutate()

            new_generation.append(child)

        self.birds = new_generation

        self.generation_id += 1
        window.replay()


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


# noinspection PyAbstractClass
class GameWindow(Window):
    def __init__(self):
        super().__init__()
        self.set_size(Const.WIDTH, Const.HEIGHT)
        self.set_caption('Floppy Bird')

        self.frame_counter = 0
        self.score = 0
        self.in_game = True

        self.bird = Bird()
        self.tubes = [Tube()]
        self.background = Sprite(img=Images.BACKGROUND_IMG)
        self.land = Land()
        self.score_label = Label(text=f'Score: {self.score}', x=Const.WIDTH - 10, y=Const.HEIGHT - 20,
                                 anchor_x='right', bold=True)
        self.game_over_label = Label(text='Game Over!', x=Const.WIDTH // 2, y=Const.HEIGHT // 2 + 20,
                                     bold=True, anchor_x='center', anchor_y='center', font_size=30)
        self.replay_label = Label(text='Press ENTER to play again.', font_size=20,
                                  x=Const.WIDTH//2, y=Const.HEIGHT//2 - 30, bold=True,
                                  anchor_x='center', anchor_y='center')

        pyglet.clock.schedule_interval(self.on_timer, Const.FRAME_RATE)

    def on_draw(self):
        self.clear()
        self.background.draw()
        for tube in self.tubes:
            tube.draw()
        self.land.draw()
        self.score_label.draw()
        self.bird.draw()
        if not self.in_game:
            self.game_over_label.draw()
            self.replay_label.draw()

    def on_timer(self, dt):
        self.frame_counter += 1

        self.check_collisions()
        self.bird.animate(dt)

        for tube in self.tubes:
            tube.animate(dt)
        self.produce_remove_tubes()

    def check_collisions(self):
        bird_x1, bird_y1, bird_x2, bird_y2 = self.bird.bbox

        if bird_y1 <= self.land.y + self.land.height:
            self.bird.die()
            self.in_game = False

        tube = self.tubes[0]
        for x1, y1, x2, y2 in tube.bbox:
            if x1 <= bird_x1 <= x2 or x1 <= bird_x2 <= x2:
                if y1 <= bird_y1 <= y2 or y1 <= bird_y2 <= y2:
                    self.bird.die()
                    self.in_game = False
        if bird_x1 > tube.center and not tube.crossed and self.bird.alive:
            self.score += 1
            self.score_label.text = f'Score: {self.score}'
            tube.crossed = True

    def produce_remove_tubes(self):
        if self.frame_counter == Const.TUBE_PERIOD * Const.FRAMES_PER_SECOND:
            self.frame_counter = 0
            tube = Tube()
            tube.draw()
            self.tubes.append(tube)

        for tube in self.tubes:
            if tube.bottom_head.x + tube.bottom_head.width < 0:
                self.tubes.remove(tube)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.bird.accelerate()
        if not self.in_game:
            if symbol == key.ENTER:
                self.replay()

    def replay(self):
        self.bird = Bird()
        self.tubes = [Tube()]
        self.score = 0
        self.frame_counter = 0
        self.in_game = True


# noinspection PyAbstractClass
class GameWindowAI(Window):
    def __init__(self):
        super().__init__()
        self.set_size(Const.WIDTH, Const.HEIGHT)
        self.set_caption('Floppy Bird')

        self.frame_counter = 0
        self.score = 0
        self.in_game = True

        self.population = Population()
        self.tubes = [Tube()]
        self.background = Sprite(img=Images.BACKGROUND_IMG)
        self.land = Land()

        pyglet.clock.schedule_interval(self.on_timer, Const.FRAME_RATE)

    def on_draw(self):
        self.clear()
        self.background.draw()
        for tube in self.tubes:
            tube.draw()
        self.land.draw()
        self.population.draw()

    def on_timer(self, dt):
        self.produce_remove_tubes()
        self.population.check_collisions()
        self.population.animate(dt)

        for tube in self.tubes:
            tube.animate(dt)

    def produce_remove_tubes(self):
        self.frame_counter += 1
        if self.frame_counter == Const.TUBE_PERIOD * Const.FRAMES_PER_SECOND:
            self.frame_counter = 0
            tube = Tube()
            tube.draw()
            self.tubes.append(tube)

        for tube in self.tubes:
            if tube.bottom_head.x + tube.bottom_head.width < 0:
                self.tubes.remove(tube)

    def get_closest_tube(self):
        return self.tubes[0]

    def replay(self):
        self.tubes = [Tube()]
        self.frame_counter = 0


if __name__ == '__main__':
    window = GameWindowAI()
    pyglet.app.run()
