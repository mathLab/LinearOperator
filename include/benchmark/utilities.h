#ifndef _utilities_h
#define _utilities_h

#include <ctime>

double take_time()
{
  return std::clock() / (double) CLOCKS_PER_SEC;
}

#endif // _utilities_h
