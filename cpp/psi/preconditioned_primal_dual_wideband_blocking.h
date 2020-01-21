#ifndef PSI_PRECONDITIONED_PRIMAL_DUAL_WIDEBAND_BLOCKING_H
#define PSI_PRECONDITIONED_PRIMAL_DUAL_WIDEBAND_BLOCKING_H

#include <iostream>
#include <functional>
#include <limits>
#include <vector>
#include "psi/primal_dual_wideband_blocking.h"
#include "psi/forward_backward.h"
#include "psi/types.h"

namespace psi {
namespace algorithm {

//! \brief Primal Dual method
//! \details This is a basic implementation of the primal dual method using a forward backwards algorithm.
//! \f$\min_{, y, z} f() + l(y) + h(z)\f$ subject to \f$Φx = y\f$, \f$Ψ^Hx = z\f$
//!  We are not implementing blocking or parallelism here.
template <class SCALAR> class PreconditionedPrimalDualWidebandBlocking : public PrimalDualWidebandBlocking<SCALAR> {
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
  //! Type of the convergence function
  typedef ConvergenceMatrixFunction<Scalar> t_IsConverged;
  //! Type of the convergence function
  typedef ProximalFunction<Scalar> t_Proximal;
  typedef psi::Matrix<Scalar> t_Matrix;
  //! Setups PreconditionedPrimalDualWidebandBlocking
  template <class Derived>
  PreconditionedPrimalDualWidebandBlocking(std::vector<EigenCell<Derived>> const &target, const t_uint &image_size, const Vector<Vector<Real>> &l2ball_epsilon, std::vector<std::vector<std::shared_ptr<const t_LinearTransform>>>& Phi, std::vector<std::vector<Vector<Real>>> const &Ui)
    : PrimalDualWidebandBlocking<typename Derived::Scalar>(target, image_size, l2ball_epsilon, Phi, Ui), itermax_fb_(20), relative_variation_fb_(1e-8){}
  virtual ~PreconditionedPrimalDualWidebandBlocking() {}

// Macro helps define properties that can be initialized as in
// auto pd  = PreconditionedPrimalDualWidebandBlocking<float>().prop0(value).prop1(value);
#define PSI_MACRO(NAME, TYPE)                                                                      \
  TYPE const &NAME() const { return NAME##_; }                                                     \
  PreconditionedPrimalDualWidebandBlocking<SCALAR> &NAME(TYPE const &NAME) {                               \
    NAME##_ = NAME;                                                                                \
    return *this;                                                                                  \
  }                                                                                                \
                                                                                                   \
protected:                                                                                         \
  TYPE NAME##_;                                                                                    \
                                                                                                   \
public:

  //! Maximum number of inner iterations (projection onto the ellipsoid)
  PSI_MACRO(itermax_fb, t_uint);
  //! Relative variation (stopping criterion)
  PSI_MACRO(relative_variation_fb, Real);
  
#undef PSI_MACRO

  //! Vector of preconditioners
  std::vector<Vector<Real>> const &Ui() const { return Ui_; }
  //! Sets the vector of preconditioners
  template <class T> PreconditionedPrimalDualWidebandBlocking<T> &Ui(EigenCell<T> const &Ui) {
    Ui_ = Ui;
    return *this;
  }

protected:
  //! Vector of preconditioners
  std::vector<Vector<Real>> Ui_;

  //! Iteration step
  void iteration_step(t_Matrix &out, std::vector<std::vector<t_Vector>> &residual, t_Matrix &p, t_Matrix &u, std::vector<std::vector<t_Vector>> &v, t_Matrix &x_bar, Vector<Real> &w_l21, Vector<Real> &w_nuclear) const;

  //! Checks input makes sense
  void sanity_check(t_Matrix const &x_guess, std::vector<t_Vector> const &res_guess) const {
    PrimalDualWidebandBlocking<SCALAR>::sanity_check(x_guess, res_guess);
    if(PrimalDualWidebandBlocking<SCALAR>::target().size() != Ui().size())
      PSI_THROW("target and preconditioning vector have inconsistent sizes (number of spectral bands)");
    for(int l = 0; l < PrimalDualWidebandBlocking<SCALAR>::target().size(); ++l){
        if((Ui()[l].array() <= 0.).any())
      PSI_THROW("inconsistent values in the preconditioning vector (each entry must be positive)");
    }
  }
};

template <class SCALAR>
void PreconditionedPrimalDualWidebandBlocking<SCALAR>::iteration_step(t_Matrix &out, std::vector<std::vector<t_Vector>> &residual, t_Matrix &p, t_Matrix &u, std::vector<std::vector<t_Vector>> &v, t_Matrix &x_bar, Vector<Real> &w_l21, Vector<Real> &w_nuclear) const {

  t_Matrix prev_sol = out;

  // p_t = p_t-1 + x_bar - prox_nuclear_norm(p_t-1 + x_bar)
  t_Matrix tmp = p + x_bar;
  t_Matrix p_prox(out.rows(), out.cols());
  proximal::nuclear_norm(p_prox, tmp, w_nuclear);
  p = tmp - p_prox; 
  // std::cout << p << "\n"; // ok

  // u_t = ...
  t_Matrix temp1_u(u.rows(), u.cols()); // see size of u
  for (int l = 0; l < PrimalDualWidebandBlocking<SCALAR>::target().size(); ++l){
    temp1_u.col(l) = u.col(l) + static_cast<t_Vector>(PrimalDualWidebandBlocking<SCALAR>::Psi().adjoint() * x_bar.col(l));
  }
  // std::cout << temp1_u << "\n"; // ok

  t_Matrix u_prox(u.rows(), u.cols());
  proximal::l21_norm(u_prox, temp1_u, w_l21);
  u = temp1_u - u_prox;
  t_Matrix temp2_u(out.rows(), out.cols());
  for (int l = 0; l < u.cols(); ++l){
    temp2_u.col(l) = static_cast<t_Vector>(PrimalDualWidebandBlocking<SCALAR>::Psi() * u.col(l));
  }
  // std::cout << temp2_u << "\n";
  // std::cout << u_prox << "\n"; // ok
  
  // v_t = v_t-1 + Phi*x_bar - l2ball_prox(v_t-1 + Phi*x_bar)
  t_Matrix temp2_v = t_Matrix::Zero(out.rows(), out.cols());
  for (int l = 0; l < PrimalDualWidebandBlocking<SCALAR>::target().size(); ++l){
    // auto l2ball_proximal = proximal::L2Ball<Scalar>(l2ball_epsilon_(l));
    t_Vector temp1_v(v[l].rows());
    temp1_v = v[l] + ((PrimalDualWidebandBlocking<SCALAR>::Phi()[l] * x_bar.col(l))).eval().cwiseProduct(static_cast<t_Vector>(Ui()[l])); // .cwiseQuotient(Ui())
    algorithm::ForwardBackward<Scalar> ellipsoid_prox = algorithm::ForwardBackward<Scalar>(temp1_v.cwiseQuotient(Ui()[l]), PrimalDualWidebandBlocking<SCALAR>::target()[l])
                         .Ui(Ui()[l])
                         .itermax(itermax_fb())
                         .l2ball_epsilon(PrimalDualWidebandBlocking<Real>::l2ball_epsilon(l))
                         .relative_variation(relative_variation_fb());
    v[l] = ellipsoid_prox();
    v[l] = temp1_v - v[l].cwiseProduct(static_cast<t_Vector>(Ui()[l]));
    temp2_v.col(l) = static_cast<t_Vector>(PrimalDualWidebandBlocking<SCALAR>::Phi()[l]->adjoint() * v[l]);
  }
  // std::cout << temp2_v << "\n"; // ok

  //x_t = positive orth projection(x_t-1 - tau * (sigma1 * Psi * s + sigma2 * Phi dagger * v))
  out = prev_sol - PrimalDualWidebandBlocking<SCALAR>::tau()*(PrimalDualWidebandBlocking<SCALAR>::kappa1()*p + temp2_u*PrimalDualWidebandBlocking<SCALAR>::kappa2() + temp2_v*PrimalDualWidebandBlocking<SCALAR>::kappa3());
  if(PrimalDualWidebandBlocking<SCALAR>::positivity_constraint()){
      out = psi::positive_quadrant(out);
  }
  x_bar = 2*out - prev_sol;

  // update the residual
  for (int l = 0; l < PrimalDualWidebandBlocking<SCALAR>::target().size(); ++l){
    residual[l] = (PrimalDualWidebandBlocking<SCALAR>::Phi()[l] * out.col(l)).eval() - PrimalDualWidebandBlocking<SCALAR>::target()[l];
  }
}
} /* psi::algorithm */
} /* psi */
#endif
