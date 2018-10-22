from setuptools import setup

setup(
    name='apisoptimizer',
    version='0.1.0',
    description='Artificial bee colony framework for tuning arbitrary functions',
    url='http://github.com/tjkessler/apisoptimizer',
    author='Travis Kessler',
    author_email='travis.j.kessler@gmail.com',
    license='MIT',
    packages=['apisoptimizer'],
    install_requires=['numpy', 'colorlogging'],
    zip_safe=False
)
