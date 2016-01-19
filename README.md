# Benchmark

## files:

  - **./bench_scripts/run.sh** : generate all benchmark

## benchmarks:

### eigen.cc

Compare a simple and a smart implementation of a code based 
on Eigen with a code based on LinearOperator Class.

The benchmark consists in the evaluation of **A^3** where **A** is a matrix. 

### full_matrix_01.cc

Define a full matrix and compare a raw implementation of matrix-vector multiplication with the one based on LinearOperator Class.

### full_matrix_02.cc

Define a full matrix **A** and compare a raw implementation of **(3Id + A)A** with the one based on LinearOperator Class.

### sparse_matrix_01.cc

Define a sparse matrix and compare a raw implementation of matrix-vector multiplication with the one based on LinearOperator Class.

### sparse_matrix_02.cc

Define a sparse matrix **A** and compare a raw implementation of $(3Id + A)A$ with the one based on LinearOperator Class.

### triple_add_01.cc

Compare a raw implementation of $A(x+y+z)$ with the one based on LinearOperator Class.

This code uses **vmult_add** in the raw section.

### triple_add_02.cc

Compare a raw implementation of $A(x+y+z)$ with the one based on LinearOperator Class.

This code uses **+=** in the raw section.

### Stokes

Benchmark for the LinearOperator class. This code assemble a preconditioner for
Stokes using two different approaches:
 - Assembling the matrix and performing
every operation in the usual way
 - Assembling the matrix, defining the
associated LinearOperators, and then performing all the operations using an
high-level syntax (LinearOperator). 

### step-6

Benchmark for constraints linear operator.

### step-40

Parallel version of the previous benchmark.