#ifndef PSI_OPERATORS_H
#define PSI_OPERATORS_H

#include "psi/config.h"
#include <array>
#include <memory>
#include <type_traits>
#include <Eigen/Core>
#include "psi/logging.h"
#include "psi/types.h"
#include "psi/maths.h"
#include "psi/wrapper.h"
#include "psi/linear_transform_operations.h"

namespace psi {

namespace details {
//! \brief Wraps a matrix into a function and its conjugate transpose
//! \details This class helps to wrap matrices into functions, such that we can use and store them
//! such that algorithms can refer to them.
template <class EIGEN> class MatrixToLinearTransform;
//! Wraps a tranposed matrix into a function and its conjugate transpose
template <class EIGEN> class MatrixAdjointToLinearTransform;
}

//! Joins together direct and indirect operators
 template <class VECTOR> class LinearTransform : 
   public details::WrapFunction<VECTOR>, public psi::LinearTransformOperations<VECTOR> {
public:
  //! Type of the wrapped functions
  typedef OperatorFunction<VECTOR> t_Function;



  //! Constructor
  //! \param[in] direct: function with signature void(VECTOR&, VECTOR const&) which applies a
  //!    linear operator to a vector.
  //! \param[in] indirect: function with signature void(VECTOR&, VECTOR const&) which applies a
  //!    the conjugate transpose linear operator to a vector.
  //! \param[in] sizes: 3 integer elements (a, b, c) such that if the input to linear operator is
  //!     of size N, then the output is of size (a * N) / b + c. A similar quantity is deduced for
  //!     the indirect operator.
  LinearTransform(t_Function const &direct, t_Function const &indirect,
                  std::array<t_int, 3> sizes = {{1, 1, 0}}, t_int const imsizey = -1,
				  t_int const imsizex = -1, t_int const oversample_factor = -11)
      : LinearTransform(
            direct, sizes, indirect,
            {{sizes[1], sizes[0], sizes[0] == 0 ? 0 : -(sizes[2] * sizes[1]) / sizes[0]}},
			imsizey, imsizex, oversample_factor) {
    assert(sizes[0] != 0);
}


  //! Constructor
  //! \param[in] direct: function with signature void(VECTOR&, VECTOR const&) which applies a
  //!    linear operator to a vector.
  //! \param[in] dsizes: 3 integer elements (a, b, c) such that if the input to the linear
  //!    operator is of size N, then the output is of size (a * N) / b + c.
  //! \param[in] indirect: function with signature void(VECTOR&, VECTOR const&) which applies a
  //!    the conjugate transpose linear operator to a vector.
  //! \param[in] dsizes: 3 integer elements (a, b, c) such that if the input to the indirect
  //!    linear operator is of size N, then the output is of size (a * N) / b + c.
  LinearTransform(t_Function const &direct, std::array<t_int, 3> dsizes, t_Function const &indirect,
                  std::array<t_int, 3> isizes, t_int const imsizey = -1, t_int const imsizex = -1,
				  t_int const oversample_factor = -11)
    : LinearTransform(details::wrap<VECTOR>(direct, dsizes), details::wrap<VECTOR>(indirect, isizes),
    		imsizey, imsizex, oversample_factor){}





  LinearTransform(details::WrapFunction<VECTOR> const &direct,
                  details::WrapFunction<VECTOR> const &indirect,
				  t_int const imsizey = -1, t_int const imsizex = -1,
				  t_int const oversample_factor = -11)
      : details::WrapFunction<VECTOR>(direct), indirect_(indirect), imsizey_(imsizey),
		imsizex_(imsizex), oversample_factor_(oversample_factor){}
		  
  LinearTransform(LinearTransform const &c)
    : details::WrapFunction<VECTOR>(c), indirect_(c.indirect_), imsizey_(c.imsizey_),
	  imsizex_(c.imsizex_), oversample_factor_(c.oversample_factor_) {}
		 
  LinearTransform(LinearTransform &&c)
    : details::WrapFunction<VECTOR>(std::move(c)), indirect_(std::move(c.indirect_)), imsizey_(std::move(c.imsizey_)),
	  imsizex_(std::move(c.imsizex_)), oversample_factor_(std::move(c.oversample_factor_)) {}
 

  void operator=(LinearTransform const &c) {
    details::WrapFunction<VECTOR>::operator=(c);
    indirect_ = c.indirect_;
    imsizey_ = c.imsizey_;
    imsizex_ = c.imsizex_;
    oversample_factor_ = c.oversample_factor_;
  }
  void operator=(LinearTransform &&c) {
    details::WrapFunction<VECTOR>::operator=(std::move(c));
    indirect_ = std::move(c.indirect_);
    imsizey_ = std::move(c.imsizey_);
    imsizex_ = std::move(c.imsizex_);
    oversample_factor_ = std::move(c.oversample_factor_);

  }

  //! Indirect transform
  LinearTransform<VECTOR> adjoint() const {
    return {indirect_, static_cast<details::WrapFunction<VECTOR> const &>(*this), imsizey(), imsizex(), oversample_factor()};
  }

  t_int imsizey() const{
	  return imsizey_;
  }

  t_int imsizex() const{
	  return imsizex_;
  }

  t_int oversample_factor() const{
	  return oversample_factor_;
  }


  // End of stub functions


  using details::WrapFunction<VECTOR>::operator*;
  using details::WrapFunction<VECTOR>::sizes;
  using details::WrapFunction<VECTOR>::rows;

private:
  //! Function applying conjugate transpose operator
  details::WrapFunction<VECTOR> indirect_;
  t_int imsizex_ = -1;
  t_int imsizey_ = -1;
  t_int oversample_factor_ = -11;
 
};



//! Helper function to creates a function operator
//! \param[in] direct: function with signature void(VECTOR&, VECTOR const&) which applies a
//!    linear operator to a vector.
//! \param[in] indirect: function with signature void(VECTOR&, VECTOR const&) which applies a
//!    the conjugate transpose linear operator to a vector.
//! \param[in] sizes: 3 integer elements (a, b, c) such that if the input to linear operator is
//!     of size N, then the output is of size (a * N) / b + c. A similar quantity is deduced for
//!     the indirect operator.
template <class VECTOR>
LinearTransform<VECTOR>
linear_transform(OperatorFunction<VECTOR> const &direct, OperatorFunction<VECTOR> const &indirect,
                 std::array<t_int, 3> const &sizes = {{1, 1, 0}}, t_int const imsizey = -1, t_int const imsizex = -1,
				  t_int const oversample_factor = -11) {
  return {direct, indirect, sizes, imsizey, imsizex, oversample_factor};
}



//! Helper function to creates a function operator
//! \param[in] direct: function with signature void(VECTOR&, VECTOR const&) which applies a
//!    linear operator to a vector.
//! \param[in] dsizes: 3 integer elements (a, b, c) such that if the input to the linear
//!    operator is of size N, then the output is of size (a * N) / b + c.
//! \param[in] indirect: function with signature void(VECTOR&, VECTOR const&) which applies a
//!    the conjugate transpose linear operator to a vector.
//! \param[in] dsizes: 3 integer elements (a, b, c) such that if the input to the indirect
//!    linear operator is of size N, then the output is of size (a * N) / b + c.
template <class VECTOR>
LinearTransform<VECTOR>
linear_transform(OperatorFunction<VECTOR> const &direct, std::array<t_int, 3> const &dsizes,
                 OperatorFunction<VECTOR> const &indirect, std::array<t_int, 3> const &isizes,
				 t_int const imsizey = -1, t_int const imsizex = -1,
				 t_int const oversample_factor = -11) {
  return {direct, dsizes, indirect, isizes, imsizey, imsizex, oversample_factor};
}

//! Convenience no-op function
template <class VECTOR>
LinearTransform<VECTOR> &linear_transform(LinearTransform<VECTOR> &passthrough) {
  return passthrough;
}
//! Creates a linear transform from a pair of wrappers
template <class VECTOR>
LinearTransform<VECTOR> linear_transform(details::WrapFunction<VECTOR> const &direct,
                                         details::WrapFunction<VECTOR> const &adjoint) {
  return {direct, adjoint};
}

namespace details {

template <class EIGEN> class MatrixToLinearTransform {
  //! The underlying raw matrix type
  typedef typename std::remove_const<typename std::remove_reference<EIGEN>::type>::type Raw;
  //! The matrix underlying the expression
  typedef typename Raw::PlainObject PlainMatrix;

public:
  //! The output type
  typedef
      typename std::conditional<std::is_base_of<Eigen::MatrixBase<PlainMatrix>, PlainMatrix>::value,
                                Vector<typename PlainMatrix::Scalar>,
                                Array<typename PlainMatrix::Scalar>>::type PlainObject;
  //! \brief Creates from an expression
  //! \details Expression is evaluated and the result stored internally. This object owns a
  //! copy of the matrix. It might share it with a few friendly neighbors.
  template <class T0>
  MatrixToLinearTransform(Eigen::MatrixBase<T0> const &A) : matrix(std::make_shared<EIGEN>(A)) {}
  //! Creates from a shared matrix.
  MatrixToLinearTransform(std::shared_ptr<EIGEN> const &x) : matrix(x){};

  //! Performs operation
  void operator()(PlainObject &out, PlainObject const &x) const {
#ifndef NDEBUG
    if((*matrix).cols() != x.size())
      PSI_THROW("Input vector and matrix do not match: ") << out.cols() << " columns for "
                                                           << x.size() << " elements.";
#endif
    out = (*matrix) * x;
  }
  //! \brief Returns conjugate transpose operator
  //! \details The matrix is shared.
  MatrixAdjointToLinearTransform<EIGEN> adjoint() const {
    return MatrixAdjointToLinearTransform<EIGEN>(matrix);
  }

private:
  //! Wrapped matrix
  std::shared_ptr<EIGEN> matrix;
};

template <class EIGEN> class MatrixAdjointToLinearTransform {
public:
  typedef typename MatrixToLinearTransform<EIGEN>::PlainObject PlainObject;
  //! \brief Creates from an expression
  //! \details Expression is evaluated and the result stored internally. This object owns a
  //! copy of the matrix. It might share it with a few friendly neighbors.
  template <class T0>
  MatrixAdjointToLinearTransform(Eigen::MatrixBase<T0> const &A)
      : matrix(std::make_shared<EIGEN>(A)) {}
  //! Creates from a shared matrix.
  MatrixAdjointToLinearTransform(std::shared_ptr<EIGEN> const &x) : matrix(x){};

  //! Performs operation
  void operator()(PlainObject &out, PlainObject const &x) const {
#ifndef NDEBUG
    if((*matrix).rows() != x.size())
      PSI_THROW("Input vector and matrix adjoint do not match: ") << out.cols() << " rows for "
                                                                   << x.size() << " elements.";
#endif
    out = matrix->adjoint() * x;
  }
  //! \brief Returns adjoint operator
  //! \details The matrix is shared.
  MatrixToLinearTransform<EIGEN> adjoint() const { return MatrixToLinearTransform<EIGEN>(matrix); }

private:
  std::shared_ptr<EIGEN> matrix;
};
}

//! Helper function to creates a function operator
template <class DERIVED>
LinearTransform<Vector<typename DERIVED::Scalar>>
linear_transform(Eigen::MatrixBase<DERIVED> const &A) {
  details::MatrixToLinearTransform<Matrix<typename DERIVED::Scalar>> const matrix(A);
  if(A.rows() == A.cols())
    return {matrix, matrix.adjoint()};
  else {
    t_int const gcd = details::gcd(A.cols(), A.rows());
    t_int const a = A.cols() / gcd;
    t_int const b = A.rows() / gcd;
    return {matrix, matrix.adjoint(), {{b, a, 0}}};
  }
}

//! Helper function to create a linear transform that's just the identity
template <class SCALAR> LinearTransform<Vector<SCALAR>> linear_transform_identity() {
  return {[](Vector<SCALAR> &out, Vector<SCALAR> const &in) { out = in; },
      [](Vector<SCALAR> &out, Vector<SCALAR> const &in) { out = in; }};
 
 }
}
#endif
