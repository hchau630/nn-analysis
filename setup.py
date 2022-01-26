from setuptools import setup, find_packages

setup(
    name='nn_analysis',
    version='0.1.2',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'h5py',
        'scikit-learn',
        'torch',
        'torchvision',
    ]
)