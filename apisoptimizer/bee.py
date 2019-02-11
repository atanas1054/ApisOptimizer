#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# bee.py (0.3.2)
#
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#

# Stdlib imports
from random import randint
from copy import deepcopy


class Bee:

    def __init__(self, param_dict, obj_fn_val, stay_limit, is_employer=False):
        '''
        Bee object for employer and onlooker bees

        Args:
            param_dict (dictionary): dictionary of Parameter objects
            obj_fn_val (int or float): value obtained by running the colony's
                                       objective function on the parameters
                                       found in param_dict
            stay_limit (int): how many neighboring food sources to search
                              before the current one is abandoned
            is_employer (bool): distinguishes an employer from an onlooker
        '''

        self.param_dict = param_dict
        self.fitness_score = self.__calc_fitness_score(obj_fn_val)
        self.obj_fn_val = obj_fn_val
        self.is_employer = is_employer
        self.__stay_count = 0
        self.__stay_limit = stay_limit
        self.abandon = False

    def mutate(self):
        '''
        Mutates one random parameter in self.param_dict
        Returns:
            dictionary: new param_dict with one mutated parameter
        '''

        param_to_change = list(self.param_dict.keys())[
            randint(0, len(self.param_dict) - 1)
        ]
        new_param_dict = deepcopy(self.param_dict)
        new_param_dict[param_to_change].mutate()
        return new_param_dict

    def is_better_food(self, obj_fn_val):
        '''
        Determines if a new food source is better than the current one

        Args:
            obj_fn_val (int or float): new objective function value

        Returns:
            bool: True if better, False if not
        '''

        if self.__calc_fitness_score(obj_fn_val) > self.fitness_score:
            return True
        else:
            return False

    def check_abandonment(self):
        '''
        Adds 1 to the number of times the bee has stayed at the current food
        source; if max number of stays has been reached, mark for abandonment
        '''

        self.__stay_count += 1
        if self.__stay_count > self.__stay_limit:
            self.abandon = True

    @staticmethod
    def __calc_fitness_score(obj_fn_val):
        '''
        Derive fitness score

        Args:
            obj_fn_val (int or float): value from objective funtion

        Returns:
            float: fitness score, i.e. normalized objective function value
        '''

        if obj_fn_val >= 0:
            return 1 / (obj_fn_val + 1)
        else:
            return 1 + abs(obj_fn_val)
