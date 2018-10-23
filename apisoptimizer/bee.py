#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# bee.py (0.1.1)
#
# Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#

# Stdlib imports
from random import randint
from copy import deepcopy


class Bee:

    def __init__(self, param_dict, obj_fn_val, stay_limit, is_employer=False):
        '''
        *param_dict*    -   dictionary of Parameter objects
        *obj_fn_val*    -   value obtained by objective function when run
                            with parameters in *param_dict*
        *stay_limit*    -   how many neighboring food sources to search before
                            the current one is abandoned (if no better food
                            found)
        *is_employer*   -   distinguishes an employer bee from an onlooker bee
        '''

        self.param_dict = param_dict
        self.fitness_score = self.__calc_fitness_score(obj_fn_val)
        self.is_employer = is_employer
        self.__stay_count = 0
        self.__stay_limit = stay_limit
        self.abandon = False

    def mutate(self):
        '''
        Mutates one parameter in self.param_dict, returns modified param_dict
        '''

        param_to_change = list(self.param_dict.keys())[
            randint(0, len(self.param_dict) - 1)
        ]
        new_param_dict = deepcopy(self.param_dict)
        new_param_dict[param_to_change].generate_rand_val()
        return new_param_dict

    def is_better_food(self, obj_fn_val):
        '''
        Determines if fitness score derived from *obj_fn_val* is better than
        than the bee's current fitness score for a given food source
        '''

        if self.__calc_fitness_score(obj_fn_val) > self.fitness_score:
            return True
        else:
            return False

    def check_abandonment(self):
        '''
        Determines whether to abandon the current food source if no better
        neighbors are found after len(self.param_dict) searches
        '''

        self.__stay_count += 1
        if self.__stay_count > self.__stay_limit:
            self.abandon = True

    @staticmethod
    def __calc_fitness_score(obj_fn_val):
        '''
        Derive fitness score from *obj_fn_val* obtained from objective
        function
        '''

        if obj_fn_val >= 0:
            return 1 / (obj_fn_val + 1)
        else:
            return 1 + abs(obj_fn_val)
