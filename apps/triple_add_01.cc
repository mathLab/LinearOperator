#include <deal.II/base/timer.h>
#include <deal.II/lac/full_matrix.h>
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

  timer.enter_subsection("raw");
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

  timer.enter_subsection("linear_operator");
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
}
