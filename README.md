# Benchmark

## Stokes

Benchmark for the LinearOperator class.
This code assemble a preconditioner for Stokes using two different approaches:
  - Assembling the matrix and performing every operation in the usual way
  - Assembling the matrix, defining the associated LinearOperators, and then performing all the operations using an high-level syntax (LinearOperator). 

## step-6

Benchmark for constraints linear operator.

## step-40

Parallel version of the previous benchmark.