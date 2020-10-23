#ifndef PSI_POWER_METHOD_H
#define PSI_POWER_METHOD_H

#include "psi/config.h"
#include <functional>
#include <limits>
#include "psi/exception.h"
#include "psi/linear_transform.h"
#include "psi/logging.h"
#include "psi/types.h"

namespace psi {
namespace algorithm {


template <class T>
  t_real power_method(std::shared_ptr<const LinearTransform<T>> const &op, const t_uint &niters,
                    const t_real &relative_difference, const T &initial_vector) {
  /* returns the sqrt of the largest eigen value of a linear operator composed with its adjoint 
     niters:: max number of iterations relative_difference::percentage difference at which eigen 
     value has converged */
  if(niters <= 0)
    return 1;
  t_real estimate_eigen_value = 1;
  t_real old_value = 0;
  T estimate_eigen_vector = initial_vector;
  estimate_eigen_vector = estimate_eigen_vector / estimate_eigen_vector.matrix().norm();
  PSI_DEBUG("Starting power method");
  PSI_DEBUG(" -[PM] Iteration: 0, norm = {}", estimate_eigen_value);
  for(t_int i = 0; i < niters; ++i) {
    estimate_eigen_vector = op->adjoint() * (*op * estimate_eigen_vector).eval();
    estimate_eigen_value = estimate_eigen_vector.matrix().norm();
    PSI_DEBUG("Iteration: {}, norm = {}", i + 1, estimate_eigen_value);
    if(estimate_eigen_value <= 0)
      throw std::runtime_error("Error in operator.");
    if(estimate_eigen_value != estimate_eigen_value)
      throw std::runtime_error("Error in operator or data corrupted.");
    estimate_eigen_vector = estimate_eigen_vector / estimate_eigen_value;
    if(relative_difference * relative_difference
       > std::abs(old_value - estimate_eigen_value) / old_value) {
      old_value = estimate_eigen_value;
      PSI_DEBUG("Converged to norm = {}, relative difference < {}", std::sqrt(old_value),
                 relative_difference);
      break;
    }
    old_value = estimate_eigen_value;
  }
  return std::sqrt(old_value);
}

template <class T>
  t_real power_method(const LinearTransform<T> &op, const t_uint &niters,
		      const t_real &relative_difference, const T &initial_vector){
  return power_method(std::make_shared<const LinearTransform<T>>(op), niters, relative_difference, initial_vector);
 }


//! \brief Eigenvalue and eigenvector for eigenvalue with largest magnitude
template <class SCALAR> class PowerMethod {
public:
  //! Scalar type
  typedef SCALAR value_type;
  //! Scalar type
  typedef value_type Scalar;
  //! Real type
  typedef typename real_type<Scalar>::type Real;
  //! Type of then underlying vectors
  typedef Vector<Scalar> t_Vector;
  //! Type of the Ψ and Ψ^H operations, as well as Φ and Φ^H
  typedef LinearTransform<t_Vector> t_LinearTransform;

  //! Holds result vector as well
  struct DiagnosticAndResult {
    //! Number of iterations
    t_uint niters;
    //! Wether convergence was achieved
    bool good;
    //! Magnitude of the eigenvalue
    Scalar magnitude;
    //! Corresponding eigenvector if converged
    Vector<Scalar> eigenvector;
  };

  PowerMethod() : itermax_(std::numeric_limits<t_uint>::max()), tolerance_(1e-8) {}
  virtual ~PowerMethod() {}

// Macro helps define properties that can be initialized as in
#define PSI_MACRO(NAME, TYPE)                                                                      \
  TYPE const &NAME() const { return NAME##_; }                                                     \
  PowerMethod<SCALAR> &NAME(TYPE const &NAME) {                                                    \
    NAME##_ = NAME;                                                                                \
    return *this;                                                                                  \
  }                                                                                                \
                                                                                                   \
protected:                                                                                         \
  TYPE NAME##_;                                                                                    \
                                                                                                   \
public:

  //! Maximum number of iterations
  PSI_MACRO(itermax, t_uint);
  //! Convergence criteria
  PSI_MACRO(tolerance, Real);
#undef PSI_MACRO

  DiagnosticAndResult AtA(std::shared_ptr<t_LinearTransform const>&A, t_Vector const &input) const;
  //! \brief Calls the power method for A.adjoint() * A
  DiagnosticAndResult AtA(t_LinearTransform const &A, t_Vector const &input) const;

  //! \brief Calls the power method for A, with A a matrix
  template <class DERIVED>
  DiagnosticAndResult operator()(Eigen::DenseBase<DERIVED> const &A, t_Vector const &input) const;

  //! \brief Calls the power method for a given matrix-vector multiplication function
  DiagnosticAndResult operator()(OperatorFunction<t_Vector> const &op, t_Vector const &input) const;

protected:
};

template <class SCALAR>
typename PowerMethod<SCALAR>::DiagnosticAndResult
PowerMethod<SCALAR>::AtA(std::shared_ptr<t_LinearTransform const>& A, t_Vector const &input) const {
  std::weak_ptr<t_LinearTransform const> weak_A(A);
  auto const op = [weak_A](t_Vector &out, t_Vector const &input) -> void {
	auto A = weak_A.lock();
	if(A){
		out = A->adjoint() * (*A * input).eval();
	}else{
		PSI_HIGH_LOG("Problem locking weak ptr in AtA power_method.h");
	}
  };
  return operator()(op, input);
}

template <class SCALAR>
typename PowerMethod<SCALAR>::DiagnosticAndResult
PowerMethod<SCALAR>::AtA(t_LinearTransform const &A, t_Vector const &input) const {
  auto const op = [&A](t_Vector &out, t_Vector const &input) -> void {
    out = A.adjoint() * (A * input).eval();
  };
  return operator()(op, input);
}

template <class SCALAR>
template <class DERIVED>
typename PowerMethod<SCALAR>::DiagnosticAndResult PowerMethod<SCALAR>::
operator()(Eigen::DenseBase<DERIVED> const &A, t_Vector const &input) const {
  Matrix<Scalar> const Ad = A.derived();
  auto const op = [&Ad](t_Vector &out, t_Vector const &input) -> void { out = Ad * input; };
  return operator()(op, input);
}

template <class SCALAR>
typename PowerMethod<SCALAR>::DiagnosticAndResult PowerMethod<SCALAR>::
operator()(OperatorFunction<t_Vector> const &op, t_Vector const &input) const {
  PSI_INFO("Computing the upper bound of a given operator");
  t_Vector eigenvector = input.normalized();
  PSI_INFO("    - eigenvector norm {}", eigenvector.stableNorm());
  typename t_Vector::Scalar previous_magnitude = 1;
  bool converged = false;
  t_uint niters = 0;

  for(; niters < itermax() and converged == false; ++niters) {
    op(eigenvector, eigenvector);
    typename t_Vector::Scalar const magnitude
        = eigenvector.stableNorm() / static_cast<Real>(eigenvector.size());
    auto const rel_val = std::abs((magnitude - previous_magnitude) / previous_magnitude);
    converged = rel_val < tolerance();
    PSI_INFO("    - Power Method Iteration {}/{} -- norm: {}", niters, itermax(), magnitude);

    eigenvector /= magnitude;
    previous_magnitude = magnitude;
  }
  // check function exists, otherwise, don't know if convergence is meaningful
  if(not converged) {
    PSI_WARN("    -  Power Method did not converge within {} iterations", itermax());
  } else {
    PSI_INFO("    -  Power Method converged in {} of {} iterations", niters, itermax());
  }
  return DiagnosticAndResult{itermax(), converged, previous_magnitude, eigenvector.normalized()};
}
}
}
#endif
