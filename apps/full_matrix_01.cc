#include "wrappers.h"
#include "test.h"

#include <iostream>

using namespace dealii;

int main(int argc, char *argv[])
{
  if (argc != 3)
    throw ExcMessage("Invalid number of command line parameters");
  unsigned int n = std::atoi(argv[1]);
  unsigned int reps = std::atoi(argv[2]);

  // Deal.II Full Matrix
  FullMatrix<double> matrix(n);
  create_full_matrix(matrix);

  // Blaze Full Matrix
  BFullMatrixShadow Bmatrix(&matrix(0,0), n, n);
  // Eigen Full Matrix
  EFullMatrixShadow Ematrix(&matrix(0,0), n, n);

  // And create temporary vectors
  Vector<double> x(n), ref(n);

  BVector Bxx(n);
  auto &Bx = static_cast<BVector::T &>(Bxx);

  EVector Ex(n);


  TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

  // ============================================================ Start Output

  std::cout << "Case 1 - FullMatrix" << std::endl;
  std::cout << "n:    " << n << std::endl;
  std::cout << "reps: " << reps << std::endl;

  // ============================================================ deal.II RAW
  reset_vector(x);

  Vector<double> tmp(n);

  timer.enter_subsection ("dealii_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      matrix.vmult(tmp, x);
      x = tmp;
      x /= norm(x);
    }
  timer.leave_subsection();

  ref = x;

  // ============================================================ deal.II LO
  reset_vector(x);

  const auto step = linear_operator(matrix) * x;

  timer.enter_subsection ("dealii_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      step.apply(x);
      x /= norm(x);
    }
  timer.leave_subsection();

  check_vector(ref,x);

  // ============================================================ Blaze Raw
  reset_vector(Bx);

  timer.enter_subsection ("blaze_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bx = Bmatrix*Bx;
      Bx /= norm(Bx);
    }
  timer.leave_subsection();

  check_vector(ref,Bx);

  // ============================================================ Blaze LO
  reset_vector(Bx);

  const auto Bstep = blaze_lo(Bmatrix) * Bxx;

  timer.enter_subsection ("blaze_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bstep.apply(Bxx);
      Bx /= norm(Bx);
    }
  timer.leave_subsection();

  check_vector(ref,Bx);

  // ============================================================ Eigen Raw
  reset_vector(Ex);

  timer.enter_subsection ("eigen_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Ex = Ematrix*Ex;
      Ex /= norm(Ex);
    }
  timer.leave_subsection();

  check_vector(ref,Ex);

  // ============================================================ Eigen LO
  reset_vector(Ex);

  auto Estep = eigen_lo(Ematrix) * Ex;

  timer.enter_subsection ("eigen_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Estep.apply(Ex);
      Ex /= norm(Ex);
    }
  timer.leave_subsection();

  check_vector(ref,Ex);
}
