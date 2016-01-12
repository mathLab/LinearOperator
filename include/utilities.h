#ifndef utilities_h
#define utilities_h

#include <deal.II/lac/vector.h>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace dealii;

void print_non_null_components(Vector<double> &v, double precision = 0, std::string str = "")
{
  if (str!="")
    {
      std::cout << "------------------------------------" << std::endl;
      std::cout << "  " << str << std::endl;
    }
  std::cout << "------------------------------------" << std::endl;
  for (unsigned int i = 0; i < v.size(); ++i)
    {
      if ( abs(v[i]) > precision)
        std::cout << std::setw(15) << i
                  << " -----> "
                  << std::setw(15) << v[i]
                  << std::endl;
    }
  std::cout << "------------------------------------" << std::endl;
}

void compare_solution(  Vector<double> &a,
                        Vector<double> &b,
                        unsigned int n_tot,
                        double precision = 1e-6,
                        bool print_ok = false,
                        bool print_no = false,
                        std::string str = "")
{
  unsigned int differences = 0;
  std::cout << "====================================" << std::endl;
  std::cout << "  Vectors " << str << " comparison :" << std::endl;
  std::cout << "------------------------------------" << std::endl;
  for (unsigned int i =0; i< n_tot; ++i)
    {
      if ( abs(b[i] - a[i]) < precision)
        {
          if (print_ok)
            {
              std::cout << std::setw(5) << i
                        << std::setw(15) << a[i]
                        << std::setw(15) << b[i];
              std::cout << std::setw(15) << " ";
              std::cout << std::endl;
            }
        }
      else
        {
          if (print_no)
            {
              std::cout << std::setw(5) << i
                        << std::setw(15) << a[i]
                        << std::setw(15) << b[i]
                        << std::setw(15) << a[i]/b[i];
              std::cout << std::setw(15) << "<-";
              std::cout << std::endl;
            }
          differences++;
        }

    }
  std::cout << "------------------------------------" << std::endl;
  std::cout << "  Precision : " << precision << std::endl;
  std::cout << "  Number of differences : " << differences << std::endl;
  std::cout << "====================================" << std::endl;
}

#endif
