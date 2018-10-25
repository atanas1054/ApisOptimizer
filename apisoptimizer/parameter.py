#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# parameter.py (0.2.0)
#
# Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#

# Stdlib imports
from random import randint, uniform

SUPPORTED_DTYPES = {
    int: randint,
    float: uniform
}


class Parameter:

    def __init__(self, name, min_val, max_val):
        '''
        *name*      -   name of the parameter
        *min_val*   -   minimum value allowed for the parameter
        *max_val*   -   maximum value allowed for the parameter
        '''

        self.value = None
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
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

        self.value = SUPPORTED_DTYPES[self.dtype](self.min_val, self.max_val)
