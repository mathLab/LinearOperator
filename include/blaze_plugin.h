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

  void reinit(const BVector &v, bool fast) {
    this->resize(v.size(), fast);
    if(fast == false)
      *this *= 0.0;
  };

  typedef double value_type;

  typedef  blaze::DynamicVector<double> T;
};

#endif
