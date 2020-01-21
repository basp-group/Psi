#ifndef SORT_UTILS
#define SORT_UTILS

#include <type_traits>
#include <algorithm>
#include <iterator>
#include <numeric> // for std::iota
#include <Eigen/Core>
#include "psi/types.h"

namespace psi {

template <typename T>
void sort_indices(const Eigen::MatrixBase<T> &v, Eigen::Ref<Eigen::Matrix<size_t, Eigen::Dynamic, 1>> idx) 
//! Returns the set of indices idx such that [vs, idx] = sort(v) (using the MATLAB definition of the sort function). With MATLAB notations, idx is such that vs = v(idx)). 
{

  // initialize original index locations
  std::iota(idx.data(), idx.data() + idx.size(), 0);

  // sort indexes based on comparing values in v
  std::sort(idx.data(), idx.data() + idx.size(),
       [&v](size_t i1, size_t i2) {return v(i1) < v(i2);});

  return;
}

template<class InputIterator, class UnaryPredicate>
  int find_if_index(InputIterator first, InputIterator last, UnaryPredicate pred)
  //! Returns the first index such that the condition is true, and -1 if the condition is never satisfied.
{
  int count = 0;
  while (first!=last) {
    if (pred(*first)){
      return count;
    } 
    ++first;
    ++count;
  }
  if(count == 0) // condition never satisfied
    count = -1;
  return count;
}

template <class ForwardIterator, class T>
  int lower_bound_index(ForwardIterator first, ForwardIterator last, const T& value)
  //! Returns the position (index) of the first element in the range [first, last) that is greater or equal to (>=) value, and -1 if no such element is found.
  //! The input vector needs to be ordered (non-increasing order) prior to applying this function.
{
  // Returns an iterator pointing to the first element in the range [first, last) that is not less than (i.e. greater or equal to) value, or last if no such element is found. 
  ForwardIterator lowerb = std::lower_bound(first, last, value);

  // Last element is > value, so none is <= value
  if(lowerb == last)
    return -1;
  else
    return lowerb - first;
}

template <class ForwardIterator, class T>
  int upper_bound_index(ForwardIterator first, ForwardIterator last, const T& value)
  //! Returns the position (index) of the last element in the range [first, last) that is lower or equal to value, and -1 if no such element is found.
  //! The input vector needs to be ordered (non-increasing order) prior to applying this function.
{
  // Returns an iterator pointing to the first element in the range [first, last) that is greater than (>) value, or last if no such element is found. 
  ForwardIterator upperb = std::upper_bound(first, last, value);

  // First element is > value, so none is <= value
  if(upperb == first)
    return -1;
  else
    return upperb - first - 1; // the -1 gives the index of the last element <= value
}

template <class ForwardIterator, class T>
  int strict_upper_bound_index(ForwardIterator first, ForwardIterator last, const T& value)
  //! Returns the position (index) of the last element in the range [first, last) that is lower than (<) value, and -1 if no such element is found.
  //! The input vector needs to be ordered (non-increasing order) prior to applying this function.
{
  ForwardIterator upperb = std::lower_bound(first, last, value);

  // First element is >= value, so none is < value
  if(upperb == first)
    return -1;
  else
    return upperb - first - 1;
}
} // end namespace psi

#endif