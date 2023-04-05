""" segfast is a library for interacting with SEG-Y files at high speed. """
import re
from setuptools import setup, find_packages


with open('segfast/__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)


with open('README.md', 'r') as f:
    long_description = f.read()


setup(
    name='segfast',
    packages=find_packages(),
    version=version,
    url='https://github.com/analysiscenter/segfast',
    license='Apache License 2.0',
    author='Sergey Tsimfer',
    author_email='sergeytsimfer@gmail.com',
    description='A library for interacting with SEG-Y seismic data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
    platforms='any',
    install_requires=[
        'dill>=0.3.1.1',
        'numpy>=1.20.0',
        'pandas>=1.0.0',
        'numba>=0.54.0',
        'segyio>=1.9.0',
    ],
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],

)