# LinearOperator - a generic, high-level expression syntax for linear algebra

[![DOI](https://zenodo.org/badge/4899/mathLab/LinearOperator.svg)](https://zenodo.org/badge/latestdoi/4899/mathLab/LinearOperator)

This repository contains all benchmarks and source codes to reproduce the 
results of the article

*LinearOperator - a generic, high-level expression syntax for linear algebra*

2016, Matthias Maier, Mauro Bardelloni, Luca Heltai

In particular, it requires the deal.II library (version greater or equal than 8.3)
and the eigen and blaze libraries.

Submodules for both eigen and blaze are provided, for your convenience. If you install
from github, you should make sure you download all submodules:

    git clone --recursive https://github.com/mathLab/LinearOperator.git

or, if you already have a clone of the LinearOperator repository, you can get the 
submodules by the following command (inside your local LinearOperator repository)

    git submodule update --init --recursive

## Installation

Assuming you installed correctly the `deal.II` library (www.dealii.org) under the path 
`/path/to/dealii`, then issuing the following commands should create all executables 
for the benchmarks presented in the paper: 

    mkdir build
    cd build
    cmake -DDEAL_II_DIR=/path/to/dealii .. # Alternatively, set DEAL_II_DIR env variable
    make 

this will create a few executables in the `build` directory.

If you want to reproduce the results of the articles, we provide the following script
    
    ./script/matrix.sh

which will create a directory `build_matrix_test`, configure, build and run all tests, 
outputting the results into the directory `./build_matrix_test/output_dir`

## benchmarks:

Each benchmark is included in the directory `./apps`, and makes a comparison between
some simple operations involving either dense matrices or sparse matrices implemented 
in `dealii`, `eigen`, or `blaze`, and two variants involving `LinearOperator` and 
`PackagedOperation` objects. The first variant involves the creation of temporary 
`PackagedOperation` objects in each loop, while the second makes one construction 
outside the loop, and only uses the objects that have been previously constructed. 

Each benchmark requires two arguments at commandline: a matrix size (for the dense
matrix case) or a refinement level (for the sparse matrix cases) and a number of 
repetitions for the test. For example

    ./full_matrix_01 100 1000

performs the test `full_matrix_01` on 100x100 matrices for 1000 repetitions, while

    ./sparse_matrix_01 3 100

performs the test `sparse_matrix_01` on a grid of 3 by 3 cells, assemblying the
stiffness matrix of a Poisson problem, and performs the test 100 times.

The following tests are provided:

- **`{full,sparse}_matrix_01.cc`**: compute `M*v`

- **`{full,sparse}_matrix_02.cc`**: compute `M*M*M*v`

- **`{full,sparse}_matrix_03.cc`**: compute `(3*Id+M)*M*v`

- **`{full,sparse}_matrix_04.cc`**: compute `M*(x+y+z)`

- **`step_32.cc`**: compute a preconditioner for Stokes system, using a low level `deal.II` implementation, 
or using a `LinearOperator` variant.
