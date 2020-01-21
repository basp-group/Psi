#ifndef PSI_PRECONDITIONED_PRIMAL_DUAL_H
#define PSI_PRECONDITIONED_PRIMAL_DUAL_H

#include "psi/primal_dual.h"
#include "psi/forward_backward.h"
#include "psi/types.h"

namespace psi {
namespace algorithm {

//! \brief Preconditioned Primal Dual method
//! \details This is a basic implementation of a primal dual algorithm involving a preconditioned proxi.
//! \f$\min_{x, y, z} f(x) + l(y) + h(z)\f$ subject to \f$Φx = y\f$, \f$Ψ^Hx = z\f$
//!  We are not implementing blocking or parallelism here.

template <class SCALAR> class PreconditionedPrimalDual : public PrimalDual<SCALAR> {
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
    typedef ConvergenceFunction<Scalar> t_IsConverged;
    //! Type of the convergence function
    typedef ProximalFunction<Scalar> t_Proximal;

    //! Setups PrimalDual
    template <class DERIVED>
    PreconditionedPrimalDual(Eigen::MatrixBase<DERIVED> const &target)
      : PrimalDual<SCALAR>(target), Ui_(Vector<Real>::Ones(target.size())), itermax_fb_(20), relative_variation_fb_(1e-8){}

    virtual ~PreconditionedPrimalDual() {}

  // Macro helps define properties that can be initialized as in
  // auto ppd  = PreconditionedPrimalDual<float>().prop0(value).prop1(value);
  #define PSI_MACRO(NAME, TYPE)                                                                     \
    TYPE const &NAME() const { return NAME##_; }                                                    \
    PreconditionedPrimalDual<SCALAR> &NAME(TYPE const &NAME) {                                      \
      NAME##_ = NAME;                                                                               \
      return *this;                                                                                 \
    }                                                                                               \
                                                                                                    \
  protected:                                                                                        \
    TYPE NAME##_;                                                                                   \
                                                                                                    \
  public:
    //! Preconditioning vector (equivalent to a diagonal preconditioner)
    PSI_MACRO(Ui, Vector<Real>);
    //! Maximum number of inner iterations (projection onto the ellipsoid)
    PSI_MACRO(itermax_fb, t_uint);
    //! Relative variation (stopping criterion)
    PSI_MACRO(relative_variation_fb, Real);
  #undef PSI_MACRO

protected:

  //! Checks input makes sense
  void iteration_step(t_Vector &out, t_Vector &residual, t_Vector &s, t_Vector &v, t_Vector &x_bar) const;

  //! Checks input makes sense
  void sanity_check(t_Vector const &x_guess, t_Vector const &res_guess) const {
    PrimalDual<SCALAR>::sanity_check(x_guess, res_guess);
    if(PrimalDual<SCALAR>::target().size() != Ui().size())
      PSI_THROW("target and preconditioning vector have inconsistent sizes");
    if((Ui().array() <= 0.).any())
      PSI_THROW("inconsistent values in the preconditioning vector (each entry must be positive)");
  }
};

template <class SCALAR>
void PreconditionedPrimalDual<SCALAR>::iteration_step(t_Vector &out, t_Vector &residual, t_Vector &s,
					     t_Vector &v, t_Vector &x_bar) const {

  t_Vector prev_sol = out;

  // v_t = v_t-1 + Ui*Phi*x_bar - Ui*ellipsoid_prox(Ui^{-1}*v_t-1 + Phi*x_bar)
  t_Vector temp = v + (PrimalDual<SCALAR>::Phi() * x_bar).eval().cwiseProduct(static_cast<t_Vector>(Ui()));
  t_Vector v_prox;
  algorithm::ForwardBackward<Scalar> ellipsoid_prox = algorithm::ForwardBackward<Scalar>(temp.cwiseQuotient(static_cast<t_Vector>(Ui())), PrimalDual<SCALAR>::target())
                         .Ui(Ui())
                         .itermax(itermax_fb())
                         .l2ball_epsilon(PrimalDual<SCALAR>::l2ball_epsilon())
                         .relative_variation(relative_variation_fb());
  v_prox = ellipsoid_prox();
  v = (temp - v_prox).cwiseProduct(static_cast<t_Vector>(Ui()));

  // s_t = s_t-1 + Psi_dagger * x_bar_t-1 - l1norm_prox(s_t-1 + Psi_dagger * x_bar_t-1)
  t_Vector temp2 = s + (PrimalDual<SCALAR>::Psi().adjoint() * x_bar);
  t_Vector s_prox;
  proximal::l1_norm(s_prox, PrimalDual<SCALAR>::kappa()/PrimalDual<SCALAR>::sigma1(), temp2);
  s = temp2 - s_prox;

  // x_t = positive orth projection(x_t-1 - tau * (sigma1 * Psi * s + sigma2 * Phi dagger * v))
  out = prev_sol - PrimalDual<SCALAR>::tau()*(PrimalDual<SCALAR>::Psi()*s*PrimalDual<SCALAR>::sigma1() + PrimalDual<SCALAR>::Phi().adjoint()*v*PrimalDual<SCALAR>::sigma2());
  if(PrimalDual<SCALAR>::positivity_constraint()){
    out = psi::positive_quadrant(out);
  }
  x_bar = 2*out - prev_sol;
  residual = (PrimalDual<SCALAR>::Phi() * out).eval() - PrimalDual<SCALAR>::target();
} /* end of preconditioned primal-dual */
} /* psi::algorithm */
} /* psi */
#endif
