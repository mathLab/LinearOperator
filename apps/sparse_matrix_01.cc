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
  Vector<double> x(n);
  
  BVector Bxx(n);
  auto &Bx = static_cast<BVector::T&>(Bxx);
  
  EVector Ex(n);

  
  TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

  // ============================================================ Start Output

  std::cout << "Case 1 - SparseMatrix" << std::endl;
  std::cout << "n:    " << n << std::endl;
  std::cout << "reps: " << reps << std::endl;

  // ============================================================ deal.II RAW
  reset_vector(x);
  
  timer.enter_subsection ("dealii_raw");
  Vector<double> tmp(n);
  for (unsigned int i = 0; i < reps; ++i)
    {
      matrix.vmult(tmp, x);
      x = tmp;
      x /= norm(x);
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << "DEBUG" << std::endl;
  std::cout << x << std::endl;
#endif
  
  // ============================================================ deal.II LO  
  reset_vector(x);

  timer.enter_subsection ("dealii_lo");
  const auto op = linear_operator(matrix);
  for (unsigned int i = 0; i < reps; ++i)
    {
      op.vmult(x, x);
      x /= norm(x);
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << x << std::endl;
#endif
  
  // ============================================================ Blaze Raw  
  reset_vector(Bx);
  
  timer.enter_subsection ("blaze_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bx = Bmatrix*Bx;
      Bx /= norm(Bx);
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << Bx << std::endl;
#endif

  // ============================================================ Blaze LO
  reset_vector(Bx);
  
  auto Blo = blaze_lo(Bmatrix);
  timer.enter_subsection ("blaze_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Blo.vmult(Bxx,Bxx);
      Bx /= norm(Bx);
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << Bx << std::endl;
#endif
  
  // ============================================================ Eigen Raw  
  reset_vector(Ex);
  
  timer.enter_subsection ("eigen_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Ex = Ematrix*Ex;
      Ex /= norm(Ex);
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << Ex << std::endl;
#endif

  // ============================================================ Eigen LO
  reset_vector(Ex);
  
  auto Elo = eigen_lo(Ematrix);
  timer.enter_subsection ("eigen_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Elo.vmult(Ex,Ex);
      Ex /= norm(Ex);
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << Ex << std::endl;
#endif
}
