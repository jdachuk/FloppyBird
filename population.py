"""
author: edacjos
created: 7/19/19
"""

import random
import os
import json
from support import Const
from player import Bird


class Population:
    def __init__(self, game):
        self.game = game
        self.individuals = []
        self.best_individual = None
        self.size = Const.POPULATION_SIZE
        self.generation_id = 0
        self.total_fitness = 0
        self.is_done = False

        self.create_birds()

    def create_birds(self):
        for _ in range(self.size):
            self.individuals.append(Bird(self.game))

    def draw(self):
        for bird in self.individuals:
            bird.draw()

    def animate(self, dt):
        for bird in self.individuals:
            bird.animate(dt)

    def check_collisions(self):
        for bird in self.individuals:
            bird.check_collisions()

        done = True
        for bird in self.individuals:
            done = done and not bird.alive

        if done:
            self.is_done = True
            self.natural_selection()

    def update_fitness(self):
        self.total_fitness = 0

        for bird in self.individuals:
            self.total_fitness += bird.fitness

    def select_best_individual(self):
        max_fitness, max_idx = 0, 0
        for idx, bird in enumerate(self.individuals):
            if bird.fitness > max_fitness:
                max_fitness = bird.fitness
                max_idx = idx

        self.best_individual = self.individuals[max_idx]
        self.save_best_individual()

    def select_individual(self):
        rand = self.total_fitness * random.random()

        cum_sum = 0
        for bird in self.individuals:
            cum_sum += bird.fitness
            if cum_sum >= rand:
                return bird

    def natural_selection(self):
        self.update_fitness()
        self.select_best_individual()

        new_generation = [self.best_individual.clone()]

        while len(new_generation) < self.size:
            parent1 = self.select_individual()
            parent2 = self.select_individual()

            child = parent1.crossover(parent2)

            child.mutate()

            new_generation.append(child)

        self.individuals = new_generation

        self.generation_id += 1
        self.is_done = False
        self.game.replay()

    def save_best_individual(self):
        try:
            os.mkdir(os.curdir + '\\data')
        except FileExistsError:
            pass
        try:
            os.mkdir(os.curdir + f'\\data\\V_{Const.VERSION}')
        except FileExistsError:
            pass
        with open(f'data\\V_{Const.VERSION}\\gen_{self.generation_id}.json', 'w') as json_file:
            data = {
                'generation_id': self.generation_id,
                'generation_size': self.size,
                'score': self.game.score,
                'total_fitness': self.total_fitness,
                'generation_avg_fitness': self.total_fitness / self.size,
                'individual_fitness': self.best_individual.fitness,
                'individual_brain': self.best_individual.brain.save_to_dict()
            }
            json.dump(data, json_file)
