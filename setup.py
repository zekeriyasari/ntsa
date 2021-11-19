from setuptools import setup

setup(
    name='ntsa',
    version='0.0.1',
    packages=['ntsa'],
    install_requires=[
        'requests',
        'importlib; python_version == "2.6"',
    ],
)
