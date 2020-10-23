#ifndef PSI_PROXIMAL_H
#define PSI_PROXIMAL_H

#include "psi/config.h"
#include <type_traits>
#include <limits> // for std::numeric_limits<double>::epsilon()
#include <Eigen/Core>
#include "psi/proximal_expression.h"
#include "psi/maths.h"

namespace psi {
//! Holds some standard proximals
namespace proximal {

//! Proximal of euclidian norm
struct EuclidianNorm {
  template <class T0>
  void operator()(Vector<typename T0::Scalar> &out,
                  typename real_type<typename T0::Scalar>::type const &t,
                  Eigen::MatrixBase<T0> const &x) const {
    typedef typename T0::Scalar Scalar;
    auto const norm = x.stableNorm();
    if(norm > t)
      out = (Scalar(1) - t / norm) * x;
    else
      out.fill(0);
  }
  //! Lazy version
  template <class T0>
  ProximalExpression<EuclidianNorm, T0>
  operator()(typename T0::Scalar const &t, Eigen::MatrixBase<T0> const &x) const {
    return {*this, t, x};
  }
};

//! Proximal of the euclidian norm
template <class T0>
auto euclidian_norm(typename real_type<typename T0::Scalar>::type const &t,
                    Eigen::MatrixBase<T0> const &x) -> decltype(EuclidianNorm(), t, x) {
  return EuclidianNorm()(t, x);
}

//! Proximal of the l1 norm
template <class T0, class T1>
void l1_norm(Eigen::DenseBase<T0> &out, typename real_type<typename T0::Scalar>::type gamma,
             Eigen::DenseBase<T1> const &x) {
  out = psi::soft_threshhold(x, gamma);
}

//! Proximal of the l1 norm
template <class T0, class T1>
void l1_norm(Eigen::DenseBase<T0> &out, Eigen::DenseBase<T1> const &gamma,
             Eigen::DenseBase<T0> const &x) {
  out = psi::soft_threshhold(x, gamma);
}

//! \brief Proximal of the l1 norm
//! \detail This specialization makes it easier to use in algorithms, e.g. within `PD::append`.
template <class S>
void l1_norm(Vector<S> &out, typename real_type<S>::type gamma, Vector<S> const &x) {
  l1_norm<Vector<S>, Vector<S>>(out, gamma, x);
}

//! \brief Proximal of l1 norm
//! \details For more complex version involving linear transforms and weights, see L1TightFrame and
//! L1 classes. In practice, this is an alias for soft_threshhold.
template <class T>
auto l1_norm(typename real_type<typename T::Scalar>::type gamma, Eigen::DenseBase<T> const &x)
    -> decltype(psi::soft_threshhold(x, gamma)) {
  return psi::soft_threshhold(x, gamma);
}

//! \brief Proximal operator of the weighted L21 norm (L2 norm along the rows)
template <class T0, class T1>
void l21_norm(Eigen::MatrixBase<T0>  &out, const Eigen::MatrixBase<T0> &x, const Eigen::MatrixBase<T1> &w) {
  auto row_norm = ((x.rowwise().norm()).array() + std::numeric_limits<typename real_type<T1>::type>::epsilon()).matrix().eval();
  out = ((row_norm - w).cwiseMax(0.).cwiseQuotient(row_norm)).asDiagonal()*x; 
}

// Distributed version (TO BE MODIFIED, ADD COMMUNICATOR...)
template <class T0, class T1>
void l21_norm_distributed(Eigen::MatrixBase<T0>  &out, const Eigen::MatrixBase<T0> &x, const Eigen::MatrixBase<T1> &w) {
  // Note: - each column of x and out is owned by a different frequency process (second dimension of x), w is a vector owned by the root frequency process
  //       - 1 b) and 2) can be directly combined [dissociated only for clarity of exposition here]
  //       - the operations below can be applied in parallel along the first dimension of x and out ("wavelet dimension" for our problem)

  // 1. Compute row-wise l2-norms for x (along frequency dimension)
  // compute squares on the workers, reduce (+) and compute sqrt on the frequency root
  
  // a) compute squares on workers and reduce (row_norm is a vector)
  auto row_norm = (x.square().rowwise().sum()).eval();

  // b) on the frequency root: compute sqrt
  row_norm = row_norm.cwiseSqrt();
  
  // 2. [frequency root] update auxiliary vector "row_norm" for the computation of the prox operator
  row_norm = (row_norm.array() + std::numeric_limits<typename real_type<T1>::type>::epsilon()).matrix().eval();
  row_norm = ((row_norm - w).cwiseMax(0.).cwiseQuotient(row_norm)); // make sure this is done in place (should be the case here)

  // 3. Broadcast "row-norm" back to each frequency worker, do the following operations there
  out = row_norm.asDiagonal()*x; 
}

//! \brief Proximal operator of the weighted L21 norm (scalar weight)
template <class T>
void l21_norm(Eigen::MatrixBase<T> &out, const Eigen::MatrixBase<T> &x, typename real_type<T>::type w) {
  auto row_norm = ((x.rowwise().norm()).array() + std::numeric_limits<typename real_type<T>::type>::epsilon()).matrix().eval();
  out = (((row_norm.array() - w).matrix()).cwiseMax(0.).cwiseQuotient(row_norm)).asDiagonal()*x;
}

//! \brief Proximal operator of the nuclear norm
template <class T0, class T1>
void nuclear_norm(Eigen::MatrixBase<T0> &out, Eigen::MatrixBase<T0> &x, Eigen::MatrixBase<T1> &w)
{
  typename Eigen::BDCSVD<typename T0::PlainObject> svd(x, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto s = psi::soft_threshhold(svd.singularValues(), w);
  out = svd.matrixU() * s.asDiagonal() * svd.matrixV().transpose();
}

template <class T>
void nuclear_norm(Eigen::MatrixBase<T> &out, Eigen::MatrixBase<T> &x, typename real_type<T>::type w)
{
  typename Eigen::BDCSVD<typename T::PlainObject> svd(x, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto s = psi::soft_threshhold(svd.singularValues(), w);
  out = svd.matrixU() * s.asDiagonal() * svd.matrixV().transpose();
}

//! Proximal for projection on the positive quadrant
template <class T>
void positive_quadrant(Vector<T> &out, typename real_type<T>::type, Vector<T> const &x) {
  out = psi::positive_quadrant(x);
};

//! Proximal for indicator function of L2 ball
template <class T> class L2Ball {
public:
  typedef typename real_type<T>::type Real;
  //! Constructs an L2 ball proximal of size epsilon
  L2Ball(Real epsilon) : epsilon_(epsilon) {}
  //! Calls proximal function
  void operator()(Vector<T> &out, typename real_type<T>::type, Vector<T> const &x) const {
    return operator()(out, x);
  }
  //! Calls proximal function
  void operator()(Vector<T> &out, Vector<T> const &x) const {
    auto const norm = x.stableNorm();
    if(norm > epsilon())
      out = x * (epsilon() / norm);
    else
      out = x;
  }
  //! Lazy version
  template <class T0>
  EnvelopeExpression<L2Ball, T0> operator()(Real const &, Eigen::MatrixBase<T0> const &x) const {
    return {*this, x};
  }

  //! Lazy version
  template <class T0>
  EnvelopeExpression<L2Ball, T0> operator()(Eigen::MatrixBase<T0> const &x) const {
    return {*this, x};
  }

  //! Size of the ball
  Real epsilon() const { return epsilon_; }
  //! Size of the ball
  L2Ball<T> &epsilon(Real eps) {
    epsilon_ = eps;
    return *this;
  }

protected:
  //! Size of the ball
  Real epsilon_;
};

template <class T> class WeightedL2Ball : public L2Ball<T> {

public:
  typedef typename L2Ball<T>::Real Real;
  typedef Vector<Real> t_Vector;
  //! Constructs an L2 ball proximal of size epsilon with given weights
  template <class T0>
  WeightedL2Ball(Real epsilon, Eigen::DenseBase<T0> const &w) : L2Ball<T>(epsilon), weights_(w) {}
  //! Constructs an L2 ball proximal of size epsilon
  WeightedL2Ball(Real epsilon) : WeightedL2Ball(epsilon, t_Vector::Ones(1)) {}
  //! Calls proximal function
  void operator()(Vector<T> &out, typename real_type<T>::type, Vector<T> const &x) const {
    return operator()(out, x);
  }
  //! Calls proximal function
  void operator()(Vector<T> &out, Vector<T> const &x) const {
    auto const norm = weights().size() == 1 ? x.stableNorm() * std::abs(weights()(0)) :
                                              (x.array() * weights().array()).matrix().stableNorm();
    if(norm > epsilon())
      out = x * (epsilon() / norm);
    else
      out = x;
  }
  //! Lazy version
  template <class T0>
  EnvelopeExpression<WeightedL2Ball, T0>
  operator()(Real const &, Eigen::MatrixBase<T0> const &x) const {
    return {*this, x};
  }
  //! Lazy version
  template <class T0>
  EnvelopeExpression<WeightedL2Ball, T0> operator()(Eigen::MatrixBase<T0> const &x) const {
    return {*this, x};
  }

  //! Weights associated with each dimension
  t_Vector const &weights() const { return weights_; }
  //! Weights associated with each dimension
  template <class T0> WeightedL2Ball<T> &weights(Eigen::MatrixBase<T0> const &w) {
    if((w.array() < 0e0).any())
      PSI_THROW("Weights cannot be negative");
    if(w.stableNorm() < 1e-12)
      PSI_THROW("Weights cannot be null");
    weights_ = w;
    return *this;
  }
  //! Size of the ball
  Real epsilon() const { return L2Ball<T>::epsilon(); }
  //! Size of the ball
  WeightedL2Ball<T> &epsilon(Real const &eps) {
    L2Ball<T>::epsilon(eps);
    return *this;
  }

protected:
  t_Vector weights_;
};

//! Translation over proximal function
template <class FUNCTION, class VECTOR> class Translation {
public:
  //! Creates proximal of translated function
  template <class T_VECTOR>
  Translation(FUNCTION const &func, T_VECTOR const &trans) : func(func), trans(trans) {}
  //! Computes proximal of translated function
  template <class OUTPUT, class T0>
  typename std::enable_if<std::is_reference<OUTPUT>::value, void>::type
  operator()(OUTPUT out, typename real_type<typename T0::Scalar>::type const &t,
             Eigen::MatrixBase<T0> const &x) const {
    func(out, t, x + trans);
    out -= trans;
  }
  //! Computes proximal of translated function
  template <class T0>
  void operator()(Vector<typename T0::Scalar> &out,
                  typename real_type<typename T0::Scalar>::type const &t,
                  Eigen::MatrixBase<T0> const &x) const {
    func(out, t, x + trans);
    out -= trans;
  }
  //! Lazy version
  template <class T0>
  ProximalExpression<Translation<FUNCTION, VECTOR> const &, T0>
  operator()(typename T0::Scalar const &t, Eigen::MatrixBase<T0> const &x) const {
    return {*this, t, x};
  }

private:
  //! Function to translate
  FUNCTION const func;
  //! Translation
  VECTOR const trans;
};

//! Translates given proximal by given vector
template <class FUNCTION, class VECTOR>
Translation<FUNCTION, VECTOR> translate(FUNCTION const &func, VECTOR const &translation) {
  return Translation<FUNCTION, VECTOR>(func, translation);
}
}
} /* psi::proximal */

#endif
