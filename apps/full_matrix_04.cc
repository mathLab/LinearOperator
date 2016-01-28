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
  Vector<double> x(n), ref(n), y(n), z(n);

  BVector Bxx(n), Byy(n), Bzz(n);
  auto &Bx = static_cast<BVector::T&>(Bxx);
  auto &By = static_cast<BVector::T&>(Byy);
  auto &Bz = static_cast<BVector::T&>(Bzz);
  
  EVector Ex(n), Ey(n), Ez(n);

  
  TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

  // ============================================================ Start Output

  std::cout << "Case 4 - FullMatrix" << std::endl;
  std::cout << "n:    " << n << std::endl;
  std::cout << "reps: " << reps << std::endl;
  
  // ============================================================ deal.II RAW
  reset_vector(x);
  y = x;
  z = x;
  
  Vector<double> tmp(n);
  
  timer.enter_subsection ("dealii_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      matrix.vmult(tmp, x);
      matrix.vmult_add(tmp, y);
      matrix.vmult_add(tmp, z);
      x = tmp;
      x /= norm(x);
    }
  timer.leave_subsection();

  ref = x;
  
  // ============================================================ deal.II LO  
  reset_vector(x);

  const auto step = linear_operator(matrix) * (x + y + z);
  
  timer.enter_subsection ("dealii_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      x = step;
      x /= norm(x);
    }
  timer.leave_subsection();

  check_vector(ref,x);

  // ============================================================ Blaze Raw
  reset_vector(Bx);
  By = Bx;
  Bz = Bx;
  
  timer.enter_subsection ("blaze_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bx = Bmatrix*(Bx+By+Bz);
      Bx /= norm(Bx);
    }
  timer.leave_subsection();

  check_vector(ref,Bx);

  // ============================================================ Blaze LO
  reset_vector(Bx);

  auto xx = PackagedOperation<BVector>(Bxx);
  auto yy = PackagedOperation<BVector>(Byy);
  auto zz = PackagedOperation<BVector>(Bzz);
  
  const auto Bstep = blaze_lo(Bmatrix) * (xx + yy + zz);
  
  timer.enter_subsection ("blaze_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bxx = Bstep;
      Bx /= norm(Bx);
    }
  timer.leave_subsection();

  check_vector(ref,Bx);

  // ============================================================ Eigen Raw
  reset_vector(Ex);
  Ey = Ex;
  Ez = Ex;
  
  timer.enter_subsection ("eigen_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Ex = Ematrix*(Ex+Ey+Ez);
      Ex /= norm(Ex);
    }
  timer.leave_subsection();

  check_vector(ref,Ex);

  // ============================================================ Eigen LO
  reset_vector(Ex);
  
  auto xxx = PackagedOperation<EVector>(Ex);
  auto yyy = PackagedOperation<EVector>(Ey);
  auto zzz = PackagedOperation<EVector>(Ez);
  
  const auto Estep = eigen_lo(Ematrix) * (xxx + yyy + zzz);
  
  timer.enter_subsection("eigen_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Ex = Estep;
      Ex /= Ex.norm();
    }
  timer.leave_subsection();

  check_vector(ref,Ex);
}
