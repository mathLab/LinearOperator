#ifndef __blaze_plugin_h
#define __blaze_plugin_h

#include <blaze/Math.h>

using blaze::StaticVector;
using blaze::DynamicVector;

class BVector : public  blaze::DynamicVector<double>
{
public:
  BVector() {};

  BVector(unsigned int n) :
    DynamicVector<double>(n) {};

  size_t memory_consumption()
  {
    return sizeof(*this);
  };

  typedef double value_type;

  typedef  blaze::DynamicVector<double> T;
};

#endif
