"""
author: edacjos
created: 7/15/19
"""

from pyglet.window import Window
from pyglet.text import Label
from game_objects import *
from population import Population


# noinspection PyAbstractClass
class GameWindowAI(Window):
    def __init__(self):
        super().__init__()
        self.set_size(Const.WIDTH, Const.HEIGHT)
        self.set_caption('Floppy Bird')

        self.frame_counter = 0
        self.score = 0

        self.population = Population(self)
        self.tubes = [Tube()]
        self.background = Sprite(img=Images.BACKGROUND_IMG)
        self.land = Land()

        self.score_label = Label(text=f'Score: {self.score}', x=Const.WIDTH - 10, y=Const.HEIGHT - 20,
                                 anchor_x='right', bold=True)

        pyglet.clock.schedule_interval(self.on_timer, Const.FRAME_RATE)

    def on_draw(self):
        self.clear()
        self.background.draw()
        for tube in self.tubes:
            tube.draw()
        self.land.draw()
        self.population.draw()
        self.score_label.draw()

    def on_timer(self, dt):
        self.produce_remove_tubes()
        self.population.check_collisions()
        self.population.animate(dt)

        for tube in self.tubes:
            tube.animate(dt)
        self.update_score()

    def update_score(self):
        tube = self.get_closest_tube()

        if Const.BIRD_X > tube.center and not tube.crossed and not self.population.is_done:
            self.score += 1
            self.score_label.text = f'Score: {self.score}'
            tube.crossed = True

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
        for tube in self.tubes:
            if tube.bbox[1][2] > 100:
                return tube

    def replay(self):
        self.tubes = [Tube()]
        self.frame_counter = 0
        self.score = 0
        self.score_label.text = f'Score: {self.score}'


if __name__ == '__main__':
    window = GameWindowAI()
    pyglet.app.run()
