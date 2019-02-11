#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# parameter.py (0.3.2)
#
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#

# Stdlib imports
from random import randint, uniform
from copy import deepcopy

SUPPORTED_DTYPES = {
    int: randint,
    float: uniform
}


class Parameter:

    def __init__(self, name, min_val, max_val, restrict):
        '''
        Parameter object: houses information for a parameter added to a Colony

        Args:
            name (str): name of the paramter
            min_val (int or float): minimum value allowed for the parameter
            max_val (int or float): maximum value allowed for the parameter
        '''

        self.value = None
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.restrict = restrict
        if type(min_val) != type(max_val):
            raise ValueError('Supplied min_val is not the same type as\
                             supplied max_val: {}, {}'.format(
                                 type(min_val),
                                 type(max_val))
                             )
        self.dtype = type(min_val + max_val)
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError('Unsupported data type: use {}'
                             .format(SUPPORTED_DTYPES))

    def generate_rand_val(self):
        '''
        Generates a random value for the parameter between min_val and max_val
        '''

        self.value = self.__randval()

    def mutate(self):
        '''
        Mutate parameter (find neighbor)
        '''

        curr_val = deepcopy(self.value)
        self.value = self.dtype(
            self.value + uniform(-1, 1) * (self.value - self.__randval())
        )
        if self.restrict:
            if self.value > self.max_val:
                self.value = self.max_val
            elif self.value < self.min_val:
                self.value = self.min_val
            if self.value == curr_val:
                self.value = curr_val
                self.mutate()

    def __randval(self):
        '''
        Calculate random value
        '''

        return SUPPORTED_DTYPES[self.dtype](self.min_val, self.max_val)
