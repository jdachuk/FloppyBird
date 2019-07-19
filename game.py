"""
author: JDachuk
created: 7/15/19
"""

from pyglet.window import Window, FPSDisplay
from pyglet.text import Label
from game_objects import *
from population import Population


# noinspection PyAbstractClass
class GameWindowAI(Window):
    def __init__(self):
        super().__init__()
        self.set_size(Const.WIDTH, Const.HEIGHT)
        self.set_caption('Floppy Bird')

        self.score = 0

        self.population = Population(self)
        self.tubes = [Tube()]
        self.background = Sprite(img=Images.BACKGROUND_IMG)
        self.land = Land()

        self.score_label = Label(text=f'Score: {self.score}', x=Const.WIDTH - 10, y=Const.HEIGHT - 20,
                                 anchor_x='right', bold=True)
        self.fps_display = FPSDisplay(self)
        self.fps_display.label.font_size = 12
        self.fps_display.label.color = (0, 255, 0, 255)
        self.fps_display.label.bold = False

        pyglet.clock.schedule_interval(self.on_timer, Const.FRAME_RATE)

    def on_draw(self):
        self.clear()
        self.background.draw()
        for tube in self.tubes:
            tube.draw()
        self.land.draw()
        self.population.draw()
        self.score_label.draw()
        self.fps_display.draw()

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
        tube = self.tubes[0]
        if tube.half_crossed:
            tube = self.tubes[1]
        if tube.bbox[0][0] < Const.WIDTH / 2 and not tube.half_crossed:
            tube.half_crossed = True
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
        self.score = 0
        self.score_label.text = f'Score: {self.score}'


if __name__ == '__main__':
    window = GameWindowAI()
    pyglet.app.run()
