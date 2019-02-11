# ApisOptimizer: Artificial Bee Colony for Tuning Function Parameters

[![GitHub version](https://badge.fury.io/gh/tjkessler%2FApisOptimizer.svg)](https://badge.fury.io/gh/tjkessler%2FApisOptimizer)
[![PyPI version](https://badge.fury.io/py/apisoptimizer.svg)](https://badge.fury.io/py/apisoptimizer)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/TJKessler/ApisOptimizer/master/LICENSE.txt)

ApisOptimizer is an open source Python package used to tune parameters for user-supplied functions. Inspired by [artificial bee colonies](https://inis.iaea.org/collection/NCLCollectionStore/_Public/41/115/41115772.pdf), ApisOptimizer is able to optimize variables in a multidimensional search space to minimize a "cost" (e.g. an error value returned by the user-supplied function).

# Installation:

### Prerequisites:
- Have Python 3.X installed
- Have the ability to install Python packages

### Method 1: pip
If you are working in a Linux/Mac environment:
```
sudo pip install apisoptimizer
```

Alternatively, in a Windows or virtualenv environment:
```
pip install apisoptimizer
```

To install with support for console/file logging:
```
pip install apisoptimizer[with_colorlogging]
```

Note: if multiple Python releases are installed on your system (e.g. 2.7 and 3.6), you may need to execute the correct version of pip. For Python 3.6, change **"pip install apisoptimizer"** to **"pip3 install apisoptimizer"**.

### Method 2: From source
- Download the ApisOptimizer repository, navigate to the download location on the command line/terminal, and execute:
```
python setup.py install
```

Additional package dependencies (ColorLogging, NumPy) will be installed during the ApisOptimizer installation process.

To update your version of ApisOptimizer to the latest release version, use:
```
pip install --upgrade apisoptimizer
```

# Usage:

To start using ApisOptimizer, you need a couple items:
- a cost function (objective function) to optimize
- parameters used by the cost function

For example, let's define a cost function to minimize the sum of three integers:

```python
def minimize_integers(integers, args=None):

    return (
        integers['int1'].value +
        integers['int2'].value +
        integers['int3'].value
    )

```

The first argument of your function must accept a **dictionary** from ApisOptimizer. The dictionary values represent the current "food source" being exploited by a given bee. Numerical values are accessed with the ".value" suffix. Dictionary keys are parameter names we supply to ApisOptimizer (more on this later).

The "args" argument acts as a "pass-through" for additional user-specified arguments. The form of this argument can vary depending on the application, but it should be either a dictionary, a list or a single value. These arguments are specified during the creation of the colony. If no additional arguments are needed, set the argument's default value to None.

Now that we have our cost function, let's import the Colony object from ApisOptimizer, initialize the colony, and add our parameters:

```python
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
```

Here we initialize the colony with 10 employer bees, supply our cost function and add our parameters. Parameters are added with a name and minimum/maximum values for its search space. By default, parameter mutations (searching a neighboring food source) will not exceed the specified parameter bounds; if this limitation is not desired, supply the "restrict=False" argument:

```python
abc.add_param('int1', 0, 10, restrict=False)
```

To add additional "pass-through" arguments, specify them with:

```python
abc = Colony(10, minimize_integers, obj_fn_args={'arg1': val1, 'arg2', val2})
```

Once we have created our colony and added our parameters, we then need to "initialize" the colony:

```python
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
```

Initializing the colony deploys employer bees (in this example, 10 bees) to random food sources (random parameter values are generated), their fitness is evaluated (in this example, lowest sum is better), and onlooker bees (equal to the number of employers) are deployed proportionally to neighboring food sources of well-performing bees.

We then send the colony through a predetermined of "search cycles":

```python
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
```

A search cycle consists of:
- each bee searches a neighboring food source (performs a mutation on one parameter)
- if the food source produces a better fitness than the bee's current food source, move there
- otherwise, the bee stays at its current food source
    - if the bee has stayed for (NE * D) cycles (NE = number of employers, D = dimension of the function, 3 in our example), abandon the food source
        - if the bee is an employer, go to a new random food source
        - if the bee is an onlooker, go to a food source neighboring a well-performing bee

We can access the colony's average fitness score, average cost function return value, best fitness score and best parameters at any time:

```python
print(abc.average_fitness)
print(abc.ave_obj_fn_val)
print(abc.best_fitness)
print(abc.best_parameters)
```

If you have the ColorLogging package installed, ApisOptimizer will not log process status messages to the console by default. If you would like to log when the colony is initialized, when a search cycle is conducted and when a new best-performing food source is found, import the logger and set the stream level to 'info':
```python
from apisoptimizer import logger

logger.stream_level = 'info'
```

To run the colony in 'debug' mode (logs processes such as generating random values, when bees find better food sources, and whether a bee abandons its food source), set the stream level to 'debug':

```python
from apisoptimizer import logger

logger.stream_level = 'debug'
```

If file logging is desired, specify the file logging level:

```python
from apisoptimizer import logger

logger.file_level = 'info'
```

By default, log files will be saved to a "logs" folder in your current working directory. To specify a different path:
```python
from apisoptimizer import logger

logger.file_level = 'info'
logger.log_dir = 'path\to\my\log\directory'
```

ApisOptimizer can utilize multiple CPU cores for concurrent processing:

```python
abc = Colony(10, minimize_integers, num_processes=8)
abc.num_processes = 8
```

Tying everything together, we have:

```python
from apisoptimizer import Colony
from apisoptimizer import logger

def minimize_integers(integers, args=None):

    return (
        integers['int1'].value +
        integers['int2'].value +
        integers['int3'].value
    )

logger.stream_level = 'info'
abc = Colony(10, minimize_integers)
abc.add_param('int1', 0, 10)
abc.add_param('int2', 0, 10)
abc.add_param('int3', 0, 10)
abc.initialize()
for _ in range(10):
    abc.search()
    print('\nAverage colony fitness: {}'.format(abc.average_fitness))
    print('Average return value: {}'.format(abc.ave_obj_fn_val))
    print('Best fitness: {}'.format(abc.best_fitness))
    print('Best parameters: {}\n'.format(abc.best_parameters))
```

And running this script produces:

```bash
[13:31:12] [INIT] [INFO] Initializing population of size 20
[13:31:12] [UPDATE] [INFO] New best performer: 12, [('int1', 7), ('int2', 2), ('int3', 3)]
[13:31:12] [UPDATE] [INFO] New best performer: 10, [('int1', 0), ('int2', 8), ('int3', 2)]
[13:31:12] [UPDATE] [INFO] New best performer: 8, [('int1', 0), ('int2', 0), ('int3', 8)]
[13:31:12] [SEARCH] [INFO] Running search iteration
[13:31:12] [UPDATE] [INFO] New best performer: 7, [('int1', 2), ('int2', 2), ('int3', 3)]

Average colony fitness: 0.0718731309683322
Average return value: 15.25
Best fitness: 0.125
Best parameters: {'int1': 2, 'int2': 2, 'int3': 3}

[13:31:12] [SEARCH] [INFO] Running search iteration
[13:31:12] [UPDATE] [INFO] New best performer: 5, [('int1', 2), ('int2', 0), ('int3', 3)]

Average colony fitness: 0.07727287908170262
Average return value: 14.55
Best fitness: 0.16666666666666666
Best parameters: {'int1': 2, 'int2': 0, 'int3': 3}

[13:31:12] [SEARCH] [INFO] Running search iteration

Average colony fitness: 0.08393703799042675
Average return value: 13.4
Best fitness: 0.16666666666666666
Best parameters: {'int1': 2, 'int2': 0, 'int3': 3}

[13:31:12] [SEARCH] [INFO] Running search iteration
[13:31:12] [UPDATE] [INFO] New best performer: 3, [('int1', 2), ('int2', 0), ('int3', 1)]

Average colony fitness: 0.09531225346123047
Average return value: 12.2
Best fitness: 0.25
Best parameters: {'int1': 2, 'int2': 0, 'int3': 1}

[13:31:12] [SEARCH] [INFO] Running search iteration

Average colony fitness: 0.1050586861538874
Average return value: 11.2
Best fitness: 0.25
Best parameters: {'int1': 2, 'int2': 0, 'int3': 1}

[13:31:12] [SEARCH] [INFO] Running search iteration

Average colony fitness: 0.11243232859331928
Average return value: 10.45
Best fitness: 0.25
Best parameters: {'int1': 2, 'int2': 0, 'int3': 1}

[13:31:12] [SEARCH] [INFO] Running search iteration
[13:31:12] [UPDATE] [INFO] New best performer: 1, [('int1', 0), ('int2', 0), ('int3', 1)]

Average colony fitness: 0.14588915470494418
Average return value: 9.6
Best fitness: 0.5
Best parameters: {'int1': 0, 'int2': 0, 'int3': 1}

[13:31:12] [SEARCH] [INFO] Running search iteration
[13:31:12] [UPDATE] [INFO] New best performer: 0, [('int1', 0), ('int2', 0), ('int3', 0)]

Average colony fitness: 0.17261901905284258
Average return value: 9.05
Best fitness: 1.0
Best parameters: {'int1': 0, 'int2': 0, 'int3': 0}

[13:31:12] [SEARCH] [INFO] Running search iteration

Average colony fitness: 0.1873122863118993
Average return value: 8.45
Best fitness: 1.0
Best parameters: {'int1': 0, 'int2': 0, 'int3': 0}

[13:31:13] [SEARCH] [INFO] Running search iteration

Average colony fitness: 0.2611601307189543
Average return value: 7.2
Best fitness: 1.0
Best parameters: {'int1': 0, 'int2': 0, 'int3': 0}
```

To run this script yourself, head over to our [examples](https://github.com/tjkessler/ApisOptimizer/tree/master/examples) directory.

# Contributing, Reporting Issues and Other Support:

To contribute to ApisOptimizer, make a pull request. Contributions should include tests for new features added, as well as extensive documentation.

To report problems with the software or feature requests, file an issue. When reporting problems, include information such as error messages, your OS/environment and Python version.

For additional support/questions, contact Travis Kessler (travis.j.kessler@gmail.com).
