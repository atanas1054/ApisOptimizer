from apisoptimizer import Colony
from ecnet.model import MultilayerPerceptron
import ecnet.data_utils
import ecnet.error_utils
import multiprocessing as mp
import matplotlib.pyplot as plt


def optimize_ecnet(param_dict, args):

    if args['num_processes'] > 0:
        model = MultilayerPerceptron(
            save_path='./tmp/model_{}/model'.format(
                mp.current_process()._identity[0] % args['num_processes']
            )
        )
    else:
        model = MultilayerPerceptron()

    df = args['DataFrame']
    df.create_sets(True, split=[0.5, 0.3, 0.2])
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
        max_epochs=param_dict['vme'].value,
        keep_prob=param_dict['keep_prob'].value
    )

    return ecnet.error_utils.calc_rmse(
        model.use(pd.test_x),
        pd.test_y
    )


if __name__ == '__main__':

    num_processes = 8

    df = ecnet.data_utils.DataFrame('cn_model_v1.0.csv')

    args = {
        'DataFrame': df,
        'num_processes': num_processes
    }

    abc = Colony(
        50,
        optimize_ecnet,
        obj_fn_args=args,
        num_processes=num_processes,
        log_level='debug',
        log_dir='./logs'
    )

    abc.add_param('learning_rate', 0.0, 1.0)
    abc.add_param('hidden_1', 8, 36)
    abc.add_param('hidden_2', 8, 36)
    abc.add_param('vme', 100, 25000)
    abc.add_param('keep_prob', 0.001, 1.0)
    abc.initialize()
    ticks = []
    obj_fn_vals = []
    for i in range(500):
        abc.search()
        ticks.append(i)
        obj_fn_vals.append(abc.ave_obj_fn_val)
    plt.plot(ticks, obj_fn_vals)
    plt.show()
