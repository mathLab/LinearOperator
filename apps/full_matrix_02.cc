#include <deal.II/base/timer.h>
#include <deal.II/lac/full_matrix.h>

#include <deal.II/lac/vector_memory.templates.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include "wrappers.h"

using namespace dealii;

int main(int argc, char *argv[])
{
  if (argc != 3)
    throw ExcMessage("Invalid number of command line parameters");
  unsigned int n = std::atoi(argv[1]);
  unsigned int reps = std::atoi(argv[2]);

  std::cout << "n:    " << n << std::endl;
  std::cout << "reps: " << reps << std::endl;

  FullMatrix<double> matrix(n);
  Vector<double> x(n);
  for (unsigned int i = 0; i < n; ++i)
    {
      x[i] = i;
      for (unsigned int j = 0; j < n; ++j)
        matrix(i, j) = 1. + 1. / (i + 1) / (j + 1);
    }

  TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

  timer.enter_subsection("dealii_raw");
  Vector<double> tmp(n);
  for (unsigned int i = 0; i < reps; ++i)
    {
      matrix.vmult(tmp, x);
      matrix.vmult(x, tmp);
      x.add(3., tmp);
      x /= x.l2_norm();
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << "DEBUG" << std::endl;
  std::cout << x << std::endl;
#endif

  for (unsigned int i = 0; i < n; ++i)
    x[i] = i;

  timer.enter_subsection("dealii_lo");
  const auto op = linear_operator(matrix);
  const auto reinit = op.reinit_range_vector;
  const auto step = (3.0 * identity_operator(reinit) + op) * op;
  
  for (unsigned int i = 0; i < reps; ++i)
    {
      step.vmult(x, x);
      x /= x.l2_norm();
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << x << std::endl;
#endif

  // Now do the same with Eigen. Let eigen interpret both x and the
  // matrix as its own objects
  EFullMatrixShadow Ematrix(&matrix(0,0), n, n);
  EVector Ex(n);

  for (unsigned int i = 0; i < n; ++i)
    Ex[i] = i;
  
  timer.enter_subsection("eigen_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Ex = (3*Ematrix*Ex+Ematrix*(Ematrix*Ex));
      Ex /= Ex.norm();
    }
  timer.leave_subsection();
  
#ifdef DEBUG
  std::cout << Ex << std::endl << std::endl;
#endif
  // Now do the same but wrap it under a linear operator
  for (unsigned int i = 0; i < n; ++i)
    Ex[i] = i;

  auto Elo = eigen_lo(Ematrix);
  const auto Ereinit = Elo.reinit_range_vector;
  const auto Estep = (3.0 * identity_operator(Ereinit) + Elo) * Elo;
  
  timer.enter_subsection("eigen_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Estep.vmult(Ex, Ex);
      Ex /= Ex.norm();
    }
  timer.leave_subsection();
  
#ifdef DEBUG
  std::cout << Ex << std::endl << std::endl;
#endif
  

  // Again, the same with Blaze.
  BVector Bxx(n);
  auto &Bx = static_cast<BVector::T&>(Bxx);

  BFullMatrixShadow Bmatrix(&matrix(0,0), n, n);
  
  for (unsigned int i = 0; i < n; ++i)
    Bx[i] = i;
  
  timer.enter_subsection("blaze_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bx = (3*Bmatrix*Bx+Bmatrix*Bmatrix*Bx);
      Bx /= std::sqrt(blaze::trans(Bx)*Bx);
    }
  timer.leave_subsection();
  
#ifdef DEBUG
  std::cout << Bx << std::endl;
#endif

  // Again the same, but wrap it under a linear operator
  for (unsigned int i = 0; i < n; ++i)
    Bx[i] = i;
  
  auto Blo = blaze_lo(Bmatrix);
  const auto Breinit = Blo.reinit_range_vector;
  const auto Bstep = (3.0 * identity_operator(Breinit) + Blo) * Blo;
  timer.enter_subsection("blaze_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bstep.vmult(Bxx,Bxx);
      Bx /= std::sqrt(blaze::trans(Bx)*Bx);
    }
  timer.leave_subsection();
  
#ifdef DEBUG
  std::cout << Bx << std::endl;
#endif
}
