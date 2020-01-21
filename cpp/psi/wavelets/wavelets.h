#ifndef PSI_WAVELETS_WAVELETS_H
#define PSI_WAVELETS_WAVELETS_H

#include "psi/config.h"
#include "psi/types.h"
#include "psi/wavelets/direct.h"
#include "psi/wavelets/indirect.h"
#include "psi/wavelets/wavelet_data.h"

namespace psi {
namespace wavelets {

// Advance declaration so we can define the subsequent friend function
class Wavelet;

//! \brief Creates a wavelet transform object
Wavelet factory(std::string name = "DB1", t_uint nlevels = 1, t_uint start_level = 0);

//! Performs direct and indirect wavelet transforms
class Wavelet : public WaveletData {
  friend Wavelet factory(std::string name, t_uint nlevels, t_uint start_level);

protected:
  //! Should be called through factory function
  Wavelet(WaveletData const &c, t_uint nlevels, t_uint start_level = 0) : WaveletData(c), levels_(nlevels), start_level_(start_level) {}

public:
  //! Destructor
  virtual ~Wavelet() {}

// Temporary macros that checks constraints on input
#define PSI_WAVELET_MACRO_MULTIPLE(NAME)                                                          \
  if((NAME.rows() == 1 or NAME.cols() == 1)) {                                                     \
    if(NAME.size() % (1 << levels()) != 0)                                                         \
      throw std::length_error("Size of " #NAME " must number a multiple of 2^levels or 1");        \
  } else if(NAME.rows() != 1 and NAME.rows() % (1 << levels()) != 0 and start_level() == 0)                               \
    throw std::length_error("Rows of " #NAME " must number a multiple of 2^levels or 1");          \
  else if(NAME.cols() % (1 << levels()) != 0 and start_level() == 0)                                                      \
    throw std::length_error("Columns of " #NAME " must number a multiple of 2^levels");
#define PSI_WAVELET_MACRO_EQUAL_SIZE(A, B)                                                        \
  if(A.rows() != B.rows() or A.cols() != B.cols())                                                 \
    A.derived().resize(B.rows(), B.cols());                                                        \
  if(A.rows() != B.rows() or A.cols() != B.cols())                                                 \
  throw std::length_error("Incorrect size for output matrix(or could not resize)")
  //! \brief Direct transform
  //! \param[in] signal: computes wavelet coefficients for this signal. Its size must be a
  //! multiple of $2^l$ where $l$ is the number of levels. Can be a matrix (2d-transform) or a
  //! column vector (1-d transform).
  //! \return wavelet coefficients
  //! \details Supports 1 and 2 dimensional tranforms for real and complex data.
  template <class T0>
  auto direct(Eigen::ArrayBase<T0> const &signal) const
      -> decltype(direct_transform(signal, 1, *this)) {
    PSI_WAVELET_MACRO_MULTIPLE(signal);
    return direct_transform(signal, levels(), *this);
  }
  //! \brief Direct transform
  //! \param[inout] coefficients: Output wavelet coefficients. Must be of the same size and type
  //! as the input.
  //! \param[in] signal: computes wavelet coefficients for this signal. Its size must be a
  //! multiple of $2^l$ where $l$ is the number of levels. Can be a matrix (2d-transform) or a
  //! column vector
  //! (1-d transform).
  //! \details Supports 1 and 2 dimensional tranforms for real and complex data.
  template <class T0, class T1>
  auto direct(Eigen::ArrayBase<T1> &coefficients, Eigen::ArrayBase<T0> const &signal) const
      -> decltype(direct_transform(coefficients, signal, 1, *this)) {
    PSI_WAVELET_MACRO_MULTIPLE(signal);
    PSI_WAVELET_MACRO_EQUAL_SIZE(coefficients, signal);
    return direct_transform(coefficients, signal, levels(), *this);
  }
  //! \brief Direct transform
  //! \param[inout] coefficients: Output wavelet coefficients. Must be of the same size and type
  //! as the input.
  //! \param[in] signal: computes wavelet coefficients for this signal. Its size must be a
  //! multiple of $2^l$ where $l$ is the number of levels. Can be a matrix (2d-transform) or a
  //! column vector
  //! (1-d transform).
  //! \details Supports 1 and 2 dimensional tranforms for real and complex data. This version
  //! allows non-constant Eigen expressions to be passe on without the ugly `const_cast` of the
  //! cannonical approach.
  template <class T0, class T1>
  auto direct(Eigen::ArrayBase<T1> &&coefficients, Eigen::ArrayBase<T0> const &signal) const
      -> decltype(direct_transform(coefficients, signal, 1, *this)) {
    PSI_WAVELET_MACRO_MULTIPLE(signal);
    PSI_WAVELET_MACRO_EQUAL_SIZE(coefficients, signal);
    return direct_transform(coefficients, signal, levels(), *this);
  }
  //! \brief Indirect transform
  //! \param[in] coefficients: Input wavelet coefficients. Its size must be a multiple of $2^l$
  //! where $l$ is the number of levels. Can be a matrix (2d-transform) or a column vector (1-d
  //! transform).
  //! \details Supports 1 and 2 dimensional tranforms for real and complex data.
  template <class T0>
  auto indirect(Eigen::ArrayBase<T0> const &coefficients) const
      -> decltype(indirect_transform(coefficients, 1, *this)) {
    PSI_WAVELET_MACRO_MULTIPLE(coefficients);
    return indirect_transform(coefficients, levels(), *this);
  }
  //! \brief Indirect transform
  //! \param[in] coefficients: Input wavelet coefficients. Its size must be a multiple of $2^l$
  //! where $l$ is the number of levels. Can be a matrix (2d-transform) or a column vector (1-d
  //! \param[inout] signal: Reconstructed signal. Must be of the same size and type as the input.
  //! \details Supports 1 and 2 dimensional tranforms for real and complex data.
  template <class T0, class T1>
  auto indirect(Eigen::ArrayBase<T1> const &coefficients, Eigen::ArrayBase<T0> &signal) const
      -> decltype(indirect_transform(coefficients, signal, 1, *this)) {
    PSI_WAVELET_MACRO_MULTIPLE(coefficients);
    PSI_WAVELET_MACRO_EQUAL_SIZE(signal, coefficients);
    return indirect_transform(coefficients, signal, levels(), *this);
  }
  //! \brief Indirect transform
  //! \param[in] coefficients: Input wavelet coefficients. Its size must be a multiple of $2^l$
  //! where $l$ is the number of levels. Can be a matrix (2d-transform) or a column vector (1-d
  //! \param[inout] signal: Reconstructed signal. Must be of the same size and type as the input.
  //! \details Supports 1 and 2 dimensional tranforms for real and complex data.  This version
  //! allows non-constant Eigen expressions to be passe on without the ugly `const_cast` of the
  //! cannonical approach.
  template <class T0, class T1>
  auto indirect(Eigen::ArrayBase<T1> const &coeffs, Eigen::ArrayBase<T0> &&signal) const
      -> decltype(indirect_transform(coeffs, signal, 1, *this)) {
    PSI_WAVELET_MACRO_MULTIPLE(coeffs);
    PSI_WAVELET_MACRO_EQUAL_SIZE(signal, coeffs);
    return indirect_transform(coeffs, signal, levels(), *this);
  }
#undef PSI_WAVELET_MACRO_MULTIPLE
#undef PSI_WAVELET_MACRO_EQUAL_SIZE
  //! Number of levels over which to do transform
  t_uint levels() const { return levels_; }
  //! Sets number of levels over which to do transform
  void levels(t_uint l) { levels_ = l; }
  //! Starting level this process owns
  t_uint start_level() const { return start_level_; }
  //! Set the starting level
  void start_level(t_uint start_level) { start_level_ = start_level; }

protected:
  //! Number of levels in the wavelet
  t_uint levels_;
  //! Starting level for this wavelet. This is to allow the wavelet to be distributed in parallel
  //! so multiple processes can have ddifferent levels. Default is 0.
  t_uint start_level_;
};
}
}
#endif
