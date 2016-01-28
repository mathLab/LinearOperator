/**
 * Additional interface required by linear operator to function
 * properly.
 */

size_t memory_consumption()
{
  return sizeof(*this);
}

/**
 * Overload operator= to reset all entries of the Matrix to the given
 * value.
 */
inline Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols > &
operator=(const _Scalar &s)
{
  this->fill(s);
  return (*this);
}

/**
 * Reinit function.
 */
inline
void reinit(const Eigen::Matrix< _Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols > &v,
            bool fast)
{
  this->conservativeResize(v.rows(), v.cols());
  if (fast == false)
    this->fill(0.0);
}
