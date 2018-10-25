from apisoptimizer import Colony
from ecnet.model import MultilayerPerceptron
import multiprocessing as mp


def optimize_ecnet(param_dict, args):

    if args['num_processes'] > 0:
        model = MultilayerPerceptron(
            id=mp.current_process()._identity[0] % args['num_processes']
        )
    else:
        model = MultilayerPerceptron()

    df = args['DataFrame']
    df.create_sets(True, split=[0.7, 0.2, 0.1])
    pd = df.package_sets()

    model.add_layer(len(pd.learn_x[0]), 'relu')
    model.add_layer(param_dict['hidden_1'].value, 'relu')
    model.add_layer(param_dict['hidden_2'].value, 'relu')
    model.add_layer(len(pd.learn_y[0]), 'linear')
    model.connect_layers()

    model.fit_validation(
        pd.learn_x,
        pd.learn_y,
        pd.valid_x,
        pd.valid_y,
        learning_rate=param_dict['learning_rate'].value,
        max_epochs=param_dict['vme'].value
    )

    return ecnet.error_utils.calc_rmse(
        model.use(pd.test_x),
        pd.test_y
    )


if __name__ == '__main__':

    num_processes = 8

    df = ecnet.data_utils.DataFrame('my_database.csv')

    args = {
        'DataFrame': df,
        'num_processes': num_processes
    }

    abc = Colony(
        5,
        optimize_ecnet,
        obj_fn_args=args,
        num_processes=num_processes
    )

    abc.add_param('learning_rate', 0.0, 0.5)
    abc.add_param('hidden_1', 8, 36)
    abc.add_param('hidden_2', 8, 36)
    abc.add_param('vme', 500, 25000)
    abc.initialize()
    for _ in range(5):
        abc.search()
        print(abc.best_fitness)
        print(abc.best_parameters)
