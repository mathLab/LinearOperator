#include <deal.II/base/timer.h>
#include<deal.II/lac/full_matrix.h>
#include<deal.II/lac/linear_operator.h>
#include<deal.II/lac/packaged_operation.h>

using namespace dealii;

int main()
{
  for (unsigned int n = 16; n <= 32568; n *= 2)
  {
    const unsigned int iter = 1073741824 / n / n;

    std::cout << "n:    " << n << std::endl;
    std::cout << "iter: " << iter << std::endl;

    TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

    FullMatrix<double> A(n);
    Vector<double> x(n);

    for (unsigned int i = 0; i < n; ++i)
    {
      x[i] = i;
      for (unsigned int j = 0; j < n; ++j)
        A(i, j) = 1. + 1. / (i + 1) / (j + 1);
    }

    timer.enter_subsection ("raw");
    Vector<double> tmp(n);
    for (unsigned int i = 0; i < iter; ++i) {
      A.vmult(tmp, x);
      x = tmp;
      x /= x.l2_norm();
    }
    timer.leave_subsection();

    for (unsigned int i = 0; i < n; ++i)
      x[i] = i;

    timer.enter_subsection ("linear_operator");
    const auto op_A = linear_operator(A);
    for (unsigned int i = 0; i < iter; ++i) {
      x = op_A * x;
      x /= x.l2_norm();
    }
    timer.leave_subsection();
  }
}