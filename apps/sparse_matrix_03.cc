#include "wrappers.h"
#include "test.h"

#include <iostream>

using namespace dealii;

int main(int argc, char *argv[])
{
  if (argc != 3)
    throw ExcMessage("Invalid number of command line parameters");
  unsigned int refinement = std::atoi(argv[1]);
  unsigned int reps = std::atoi(argv[2]);

  // Deal.II Sparse Matrix
  SparseMatrix<double> matrix;
  create_sparse_matrix(matrix, refinement);
  unsigned int n = matrix.m();

  // Blaze Sparse Matrix
  BSparseMatrix Bmatrix;
  copy(Bmatrix, matrix);

  // Eigen Sparse Matrix
  ESparseMatrix Ematrix;
  copy(Ematrix, matrix);


  // And create temporary vectors
  Vector<double> x(n), ref(n);

  BVector Bxx(n);
  auto &Bx = static_cast<BVector::T &>(Bxx);

  EVector Ex(n);


  TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

  // ============================================================ Start Output

  std::cout << "Case 2 - SparseMatrix" << std::endl;
  std::cout << "n:    " << n << std::endl;
  std::cout << "reps: " << reps << std::endl;

  // ============================================================ deal.II RAW
  reset_vector(x);

  Vector<double> tmp(n);

  timer.enter_subsection ("dealii_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      matrix.vmult(tmp, x);
      matrix.vmult(x, tmp);
      x.add(3., tmp);
      x /= norm(x);
    }
  timer.leave_subsection();

  ref = x;

  // ============================================================ deal.II LO
  reset_vector(x);

  const auto op = linear_operator(matrix);
  const auto reinit = op.reinit_range_vector;
  const auto step = (3.0 * identity_operator(reinit) + op) * op;

  timer.enter_subsection ("dealii_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      step.vmult(x,x);
      x /= norm(x);
    }
  timer.leave_subsection();

  check_vector(ref,x);

  // ============================================================ Blaze Raw
  reset_vector(Bx);

  timer.enter_subsection ("blaze_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bx = 3*Bmatrix*Bx + Bmatrix*Bmatrix*Bx;
      Bx /= norm(Bx);
    }
  timer.leave_subsection();

  check_vector(ref,Bx);

  // ============================================================ Blaze LO
  reset_vector(Bx);

  const auto Blo = blaze_lo(Bmatrix);
  const auto Breinit = Blo.reinit_range_vector;
  const auto Bstep = (3.0 * identity_operator(Breinit) + Blo) * Blo;

  timer.enter_subsection ("blaze_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bstep.vmult(Bxx,Bxx);
      Bx /= norm(Bx);
    }
  timer.leave_subsection();

  check_vector(ref,Bx);

  // ============================================================ Eigen Raw
  reset_vector(Ex);

  timer.enter_subsection ("eigen_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Ex = 3*Ematrix*Ex + Ematrix*(Ematrix*Ex);
      Ex /= norm(Ex);
    }
  timer.leave_subsection();

  check_vector(ref,Ex);

  // ============================================================ Eigen LO
  reset_vector(Ex);

  const auto Elo = eigen_lo(Ematrix);
  const auto Ereinit = Elo.reinit_range_vector;
  const auto Estep = (3.0 * identity_operator(Ereinit) + Elo) * Elo;

  timer.enter_subsection ("eigen_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Estep.vmult(Ex,Ex);
      Ex /= norm(Ex);
    }
  timer.leave_subsection();

  check_vector(ref,Ex);
}
