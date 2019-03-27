#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# optimize_ecnet.py
#
# Example script, optimize the following neural network learning/architecture
#   hyperparameters for Cetane Number prediction:
#   - Learning rate
#   - Learning rate decay
#   - Neurons per hidden layer
#   - Beta_1 (Adam Optimizer)
#   - Beta_2 (Adam Optimizer)
#   - Epsilon (Adam Optimizer)
#

from apisoptimizer import Colony
from apisoptimizer import logger
from ecnet.utils.server_utils import default_config, train_model
from ecnet.utils.data_utils import DataFrame


def optimize_ecnet(param_dict, args):

    vars = default_config()
    vars['beta_1'] = param_dict['beta_1'].value
    vars['beta_2'] = param_dict['beta_2'].value
    vars['epsilon'] = param_dict['epsilon'].value
    vars['learning_rate'] = param_dict['learning_rate'].value
    vars['decay'] = param_dict['decay'].value
    vars['hidden_layers'][0][0] = param_dict['hidden_1'].value
    vars['hidden_layers'][1][0] = param_dict['hidden_2'].value

    dataframe = args['dataframe']
    sets = dataframe.package_sets()
    return train_model(sets, vars, 'test', 'rmse', validate=True, save=False)


if __name__ == '__main__':

    logger.stream_level = 'debug'

    dataframe = DataFrame('cn_model_v1.0.csv')
    dataframe.create_sets()

    abc = Colony(
        10,
        optimize_ecnet,
        obj_fn_args={'dataframe': dataframe},
        num_processes=4
    )

    abc.add_param('beta_1', 0.0, 1.0)
    abc.add_param('beta_2', 0.0, 1.0)
    abc.add_param('epsilon', 0.0, 1.0)
    abc.add_param('learning_rate', 0.0, 1.0)
    abc.add_param('decay', 0.0, 1.0)
    abc.add_param('hidden_1', 1, 50)
    abc.add_param('hidden_2', 1, 50)
    abc.initialize()
    for i in range(10):
        abc.search()
        print('\nAverage colony fitness: {}'.format(abc.average_fitness))
        print('Average return value: {}'.format(abc.ave_obj_fn_val))
        print('Best fitness: {}'.format(abc.best_fitness))
        print('Best parameters: {}\n'.format(abc.best_parameters))
