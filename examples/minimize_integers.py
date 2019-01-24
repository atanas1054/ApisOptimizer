from apisoptimizer import Colony


def minimize_integers(integers, args=None):

    return (
        integers['int1'].value +
        integers['int2'].value +
        integers['int3'].value
    )

abc = Colony(10, minimize_integers)
abc.add_param('int1', 0, 10)
abc.add_param('int2', 0, 10)
abc.add_param('int3', 0, 10)
abc.initialize()
for _ in range(10):
    abc.search()
    print('Average colony fitness: {}'.format(abc.average_fitness))
    print('Average return value: {}'.format(abc.ave_obj_fn_val))
    print('Best fitness: {}'.format(abc.best_fitness))
    print('Best parameters: {}'.format(abc.best_parameters))
