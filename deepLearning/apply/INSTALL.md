# Buildling and Installation Instructions

This is an overview of building and installing the software. 

## Prerequisites

While I use a variety of frameworks for machine learning their common thread is that they have C++ interfaces. 

    1.  A C++ compiler with a sufficient 2017 standards implementation.
    2.  [CMake](https://cmake.org/) v.3.12
    3.  [libtorch](https://pytorch.org/cppdocs/) v.1.50 for model evaluation.
    4.  [rtseis](https://github.com/uofuseismo/rtseis) for preprocessing.
    5.  [HDF5](https://www.hdfgroup.org/solutions/hdf5/) for reading model coefficients.
    6.  [GTest](https://github.com/google/googletest) for testing.

If you are interested in accelerating the deep learning models with a GPU then you'll have to get CUDA on your system.  I recommend backing up all your files, reading all the directions before doing anything, and having a second computer handy.


