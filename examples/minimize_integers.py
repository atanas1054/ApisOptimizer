from apisoptimizer import Colony


def minimize_integers(param_dict, args=None):

    sum = 0
    for param in param_dict:
        sum += param_dict[param].value
    return sum


if __name__ == '__main__':

    abc = Colony(20, minimize_integers, num_processes=4)
    abc.add_param('int1', 0, 10)
    abc.add_param('int2', 0, 10)
    abc.add_param('int3', 0, 10)
    abc.initialize()
    for _ in range(10):
        abc.search()
        print(abc.best_fitness)
        print(abc.best_parameters)
