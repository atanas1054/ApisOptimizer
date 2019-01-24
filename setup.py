from setuptools import setup

setup(
    name='apisoptimizer',
    version='0.3.0',
    description='Artificial bee colony framework for tuning variables in'
                ' user-suppled functions',
    url='http://github.com/tjkessler/apisoptimizer',
    author='Travis Kessler',
    author_email='travis.j.kessler@gmail.com',
    license='MIT',
    packages=['apisoptimizer'],
    install_requires=['colorlogging', 'numpy'],
    zip_safe=False
)
