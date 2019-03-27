#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# minimize_integers.py
#
# Example script, minimize the values of three integers using the artificial
#   bee colony algorithm
#

# ApisOptimizer imports
from apisoptimizer import Colony
from apisoptimizer import logger


def minimize_integers(integers, args=None):
    ''' Fitness function, to be passed to ABC. The goal of this function and
    the ABC is to minimize the sum by adjusting the values of the supplied
    integers.

    Args:
        integers (dict): dictionary of apisoptimizer.Parameter objects, i.e.
            the values to be tuned (in this case, three integers)
        args (None): there are no additional arguments for this function

    Returns:
        int: sum of three integer values, the ABC will minimize this
    '''

    return (
        integers['int1'].value +
        integers['int2'].value +
        integers['int3'].value
    )


if __name__ == '__main__':

    # Set the stream logging level to `info`
    logger.stream_level = 'info'

    # Initialize the colony with 10 employers, pass it the fitness function
    abc = Colony(10, minimize_integers)

    # Add three integer values (initialized between 0 and 10)
    abc.add_param('int1', 0, 10)
    abc.add_param('int2', 0, 10)
    abc.add_param('int3', 0, 10)

    # Initialize the ABC (create employers and onlookers)
    abc.initialize()

    # Run 10 times
    for _ in range(10):

        # Run the search algorithm
        abc.search()

        print('\nAverage colony fitness: {}'.format(abc.average_fitness))
        print('Average return value: {}'.format(abc.ave_obj_fn_val))
        print('Best fitness: {}'.format(abc.best_fitness))
        print('Best parameters: {}\n'.format(abc.best_parameters))
