from setuptools import setup

setup(
    name='apisoptimizer',
    version='0.3.2',
    description='Artificial bee colony framework for tuning variables in'
                ' user-suppled functions',
    url='http://github.com/tjkessler/apisoptimizer',
    author='Travis Kessler',
    author_email='travis.j.kessler@gmail.com',
    license='MIT',
    packages=['apisoptimizer'],
    install_requires=['numpy'],
    extras_require={'with_colorlogging': ['colorlogging']},
    zip_safe=False
)
