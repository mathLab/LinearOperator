#include <deal.II/base/timer.h>
#include <deal.II/lac/full_matrix.h>

#include "wrappers.h"

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

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
  Vector<double> y = x;
  Vector<double> z = x;

  TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

  timer.enter_subsection("dealii_raw");
  Vector<double> tmp(n);
  for (unsigned int i = 0; i < reps; ++i)
    {
      matrix.vmult(tmp, x);
      matrix.vmult_add(tmp, y);
      matrix.vmult_add(tmp, z);
      x = tmp;
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
  const auto step = linear_operator(matrix) * (x + y + z);
  for (unsigned int i = 0; i < reps; ++i)
    {
      x = step;
      x /= x.l2_norm();
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << x << std::endl;
#endif

  // ============================================================ blaze raw
  BFullMatrixShadow Bmatrix(&matrix(0,0), n, n);
  BVector Bxx(n), Byy(n), Bzz(n);
  auto &Bx = static_cast<BVector::T&>(Bxx);
  auto &By = static_cast<BVector::T&>(Byy);
  auto &Bz = static_cast<BVector::T&>(Bzz);
  
  for (unsigned int i = 0; i < n; ++i)
    Bx[i] = i;

  By = Bx;
  Bz = Bx;

  timer.enter_subsection("blaze_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bx = Bmatrix*(Bx+By+Bz);
      Bx /= std::sqrt(blaze::trans(Bx)*Bx);
    }
  timer.leave_subsection();
  
#ifdef DEBUG
  std::cout << Bx << std::endl;
#endif
  // ============================================================ blaze lo
  for (unsigned int i = 0; i < n; ++i)
    Bx[i] = i;

  auto xx = PackagedOperation<BVector>(Bxx);
  auto yy = PackagedOperation<BVector>(Byy);
  auto zz = PackagedOperation<BVector>(Bzz);
  
  
  timer.enter_subsection("blaze_lo");
  const auto Bstep = blaze_lo(Bmatrix) * (xx + yy + zz);
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bxx = Bstep;
      Bx /= std::sqrt(blaze::trans(Bx)*Bx);
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << Bx << std::endl;
#endif

  // ============================================================ eigen raw
  EFullMatrixShadow Ematrix(&matrix(0,0), n, n);
  EVector Ex(n), Ey(n), Ez(n);

  for (unsigned int i = 0; i < n; ++i)
    Ex[i] = i;

  Ey = Ex;
  Ez = Ex;

  timer.enter_subsection("eigen_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Ex = Ematrix*(Ex+Ey+Ez);
      Ex /= Ex.norm();
    }
  timer.leave_subsection();
  
#ifdef DEBUG
  std::cout << Ex << std::endl;
#endif
  // ============================================================ eigen lo
  for (unsigned int i = 0; i < n; ++i)
    Ex[i] = i;

  auto xxx = PackagedOperation<EVector>(Ex);
  auto yyy = PackagedOperation<EVector>(Ey);
  auto zzz = PackagedOperation<EVector>(Ez);
  
  
  timer.enter_subsection("eigen_lo");
  const auto Estep = eigen_lo(Ematrix) * (xxx + yyy + zzz);
  for (unsigned int i = 0; i < reps; ++i)
    {
      Ex = Estep;
      Ex /= Ex.norm();
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << Ex << std::endl;
#endif
}
