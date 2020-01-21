#ifndef PSI_PRIMAL_DUAL_H
#define PSI_PRIMAL_DUAL_H

#include "psi/config.h"
#include <functional>
#include <limits>
#include "psi/exception.h"
#include "psi/linear_transform.h"
#include "psi/logging.h"
#include "psi/types.h"
#include "psi/l1_proximal.h"
#include "psi/proximal.h"
#include "psi/utilities.h"

namespace psi {
namespace algorithm {

//! \brief Primal Dual method
//! \details This is a basic implementation of the primal dual method using a forward backwards algorithm.
//! \f$\min_{x, y, z} f(x) + l(y) + h(z)\f$ subject to \f$Φx = y\f$, \f$Ψ^Hx = z\f$
//!  We are not implementing blocking or parallelism here.
template <class SCALAR> class PrimalDual {
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
  typedef psi::Matrix<Scalar> Matrix;
  //! Values indicating how the algorithm ran
  struct Diagnostic {
    //! Number of iterations
    t_uint niters;
    //! Wether convergence was achieved
    bool good;
    //! the residual from the last iteration
    t_Vector residual;

    Diagnostic(t_uint niters = 0u, bool good = false)
        : niters(niters), good(good), residual(t_Vector::Zero(1)) {}
    Diagnostic(t_uint niters, bool good, t_Vector &&residual)
        : niters(niters), good(good), residual(std::move(residual)) {}
  };
  //! Holds result vector as well
  struct DiagnosticAndResult : public Diagnostic {
    //! Output x
    t_Vector x;
    t_Vector u;
    t_Vector v;
    t_Vector x_bar; 
    //! epsilon parameter (updated for real data)
    Real epsilon; // necessary for epsilon update
  };

  //! Setups PrimalDual
  template <class DERIVED>
  PrimalDual(Eigen::MatrixBase<DERIVED> const &target)
    : itermax_(std::numeric_limits<t_uint>::max()), is_converged_(), kappa_(1), tau_(1), sigma1_(1),
      sigma2_(1), levels_(1), nu_(1), l2ball_epsilon_(1), l1_proximal_weights_(Vector<Real>::Zero(1)),
      Phi_(linear_transform_identity<Scalar>()),
      Psi_(linear_transform_identity<Scalar>()),
      residual_convergence_(1e-4), relative_variation_(1e-4), positivity_constraint_(true),
      target_(target) {}
  virtual ~PrimalDual() {}

// Macro helps define properties that can be initialized as in
// auto pd  = PrimalDual<float>().prop0(value).prop1(value);
#define PSI_MACRO(NAME, TYPE)                                                                     \
  TYPE const &NAME() const { return NAME##_; }                                                     \
  PrimalDual<SCALAR> &NAME(TYPE const &NAME) {                                                     \
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
    //! ν parameter
  PSI_MACRO(nu, Real);
  //! κ
  PSI_MACRO(kappa, Real);
  //! τ
  PSI_MACRO(tau, Real);
  //! σ
  PSI_MACRO(sigma1, Real);
  //! ς
  PSI_MACRO(sigma2, Real);
  //! Number of dictionaries used in the wavelet operator
  PSI_MACRO(levels, t_uint);

  PSI_MACRO(l1_proximal_weights, Vector<Real>);
  
  PSI_MACRO(l2ball_epsilon, Real);
  //! A function verifying convergence
  PSI_MACRO(is_converged, t_IsConverged);
  //! Measurement operator
  PSI_MACRO(Phi, t_LinearTransform);
  //! Analysis operator
  PSI_MACRO(Psi, t_LinearTransform);
  //! Convergence of the residuals
  //! If negative it is disabled
  PSI_MACRO(residual_convergence, Real);
  //!  Convergence of the relative variation of the objective functions
  //!  If negative, this convergence criteria is disabled.
  PSI_MACRO(relative_variation, Real);
  //! Enforce whether the result needs to be projected to the positive quadrant or not
  PSI_MACRO(positivity_constraint, bool);


  
#undef PSI_MACRO

  //! Vector of target measurements
  t_Vector const &target() const { return target_; }
  //! Sets the vector of target measurements
  template <class DERIVED> PrimalDual<DERIVED> &target(Eigen::MatrixBase<DERIVED> const &target) {
    target_ = target;
    return *this;
  }

    
  
  //! Facilitates call to user-provided convergence function
  bool is_converged(t_Vector const &x) const {
    return static_cast<bool>(is_converged()) and is_converged()(x);
  }
  

  //! \brief Calls Primal Dual
  //! \param[out] out: Output vector x
  //! \param[in] guess: initial guess for x and residuals
  DiagnosticAndResult
  operator()(std::tuple<t_Vector const &, t_Vector const &> const &guess) {
    return operator()(initial_guess(guess));
  }

  
  //! \brief Calls Primal Dual
  //! \param[out] out: Output vector x
  Diagnostic operator()(t_Vector &out) { return operator()(out, initial_guess()); }
  //! \brief Calls Primal Dual
  //! \param[out] out: Output vector x
  //! \param[in] guess: initial guess
  // AJ TODO Fix comments above
  Diagnostic operator()(t_Vector &out, std::tuple<t_Vector, t_Vector, t_Vector, t_Vector, t_Vector, Real> const &guess) {
    return operator()(out, std::get<0>(guess), std::get<1>(guess), std::get<2>(guess), std::get<3>(guess), std::get<4>(guess), std::get<5>(guess));
  }
  
  //! \brief Calls Primal Dual
  //! \param[in] guess: initial guess
  //  DiagnosticAndResult operator()(DiagnosticAndResult const &guess) {
  //   DiagnosticAndResult result;
    // Need to refactor the operator to allow this to happen without the explicit copy between guess and result below.
  //   result.u = guess.u;
  //  result.v = guess.v;
  //  result.x_bar = guess.x_bar;
  //  result.epsilon = guess.epsilon;
  // static_cast<Diagnostic &>(result) = operator()(result.x, guess.x, result.u, result.v, result.x_bar, guess.residual, result.epsilon);
  //  return result;
  // }
  //! \brief Calls Primal Dual
  //! \param[in] guess: initial guess
  DiagnosticAndResult operator()() {
    DiagnosticAndResult result;
    std::tuple<t_Vector, t_Vector, t_Vector, t_Vector, t_Vector, Real> guess = initial_guess();
    result.u = std::get<1>(guess);
    result.v = std::get<2>(guess);
    result.x_bar = std::get<3>(guess);
    result.epsilon = std::get<5>(guess); 
    static_cast<Diagnostic &>(result) = operator()(result.x, std::get<0>(guess), result.u, result.v, result.x_bar,  std::get<4>(guess), result.epsilon);
// AJ I'd ideally do the below as it involves less memory operations, but I need to investigate how you do this in C++ more thoroughly first.
//    t_Vector initial_x, initial_residual;
//   (initial_x, result.s, result.v, result.x_bar, initial_residual) = initial_guess();
//   static_cast<Diagnostic &>(result) = operator()(result.x, initial_x, result.u, result.v, result.x_bar, initial_residual);
    return result;
  }
  //! Makes it simple to chain different calls to Primal Dual
  DiagnosticAndResult operator()(DiagnosticAndResult const &warmstart) {
    DiagnosticAndResult result = warmstart;
// Need to refactor the operator to allow this to happen without the explicit copy between warmstart and result below.
// As the initialisation above occurs most of below is redundant but needs checked.
    result.u = warmstart.u;
    result.v = warmstart.v;
    result.x_bar = warmstart.x_bar;
    result.epsilon = warmstart.epsilon;
    static_cast<Diagnostic &>(result) = operator()(result.x, warmstart.x, result.u, result.v, result.x_bar, warmstart.residual, result.epsilon);
    return result;
  }
  //! \brief Analysis operator Ψ and Ψ^† 
  template <class... ARGS>
  typename std::enable_if<sizeof...(ARGS) >= 1, PrimalDual &>::type Psi(ARGS &&... args) {
    Psi_ = linear_transform(std::forward<ARGS>(args)...);
    return *this;
  }


  //! Set Φ and Φ^† using arguments that psi::linear_transform understands
  template <class... ARGS>
  typename std::enable_if<sizeof...(ARGS) >= 1, PrimalDual &>::type Phi(ARGS &&... args) {
    Phi_ = linear_transform(std::forward<ARGS>(args)...);
    return *this;
  }

  //! \brief Sets up x and residual with provided data and initialises everything else
  DiagnosticAndResult initial_guess(std::tuple<t_Vector const &, t_Vector const &>  const &guess) const {
    DiagnosticAndResult created_guess;
    created_guess.x = std::get<0>(guess);
    created_guess.u = t_Vector(std::get<0>(guess).size()*levels());  // u
    created_guess.u = t_Vector(target().size());  // v
    created_guess.x_bar = t_Vector::Zero(std::get<0>(guess).size());  // x_bar
    created_guess.residual = std::get<1>(guess);
    created_guess.epsilon = l2ball_epsilon_; // epsilon
    return created_guess;
  }


  //! \brief Computes initial guess for x and the residual
  //! \details with y the vector of measurements
  //! - x = Φ^T y
  //! - residuals = Φ x - y
  std::tuple<t_Vector, t_Vector, t_Vector, t_Vector, t_Vector, Real> initial_guess() const {
    std::tuple<t_Vector, t_Vector, t_Vector, t_Vector, t_Vector, Real> guess;
    // AJ TODO REplace the below with a size passed in to the algorithm rather than having to do a 
    // calculation to get the size.
    std::get<0>(guess) = t_Vector::Zero((Phi().adjoint()*target()).size()); // x
    std::get<1>(guess) = t_Vector(std::get<0>(guess).size()*levels());  // u
    std::get<2>(guess) = t_Vector(target().size());  // v
    std::get<3>(guess) = t_Vector::Zero(std::get<0>(guess).size());  // x_bar
    std::get<4>(guess) = (Phi() * std::get<0>(guess)).eval() - target(); // residual
    std::get<5>(guess) = l2ball_epsilon_; // epsilon
    return guess;
  }


protected:
  //! Vector of measurements
  t_Vector target_;
  
  void iteration_step(t_Vector &out, t_Vector &residual, t_Vector &u, t_Vector &v, t_Vector &x_bar) const;

  //! Checks input makes sense
  void sanity_check(t_Vector const &x_guess, t_Vector const &u_guess, t_Vector const &v_guess, t_Vector const &x_bar_guess, t_Vector const &res_guess, Real const &l2ball_epsilon_guess) const {
    if((Phi().adjoint() * target()).size() != x_guess.rows())
      PSI_THROW("target, adjoint measurement operator and input vector have inconsistent sizes");
    if(x_bar_guess.size() != x_guess.size())
      PSI_THROW("x_bar and x have inconsistent sizes");
    if(target().size() != res_guess.size())
      PSI_THROW("target and residual vector have inconsistent sizes");
    if(u_guess.size() != x_guess.size()*levels())
      PSI_THROW("input vector, measurement operator and dual variable u have inconsistent sizes");
    if(v_guess.size() != target().size())
      PSI_THROW("target and dual variable v have inconsistent sizes");
    if(not static_cast<bool>(is_converged()))
      PSI_WARN("No convergence function was provided: algorithm will run for {} steps", itermax());
  }

  //! \brief Calls Primal Dual
  //! \param[out] out: Output vector x
  //! \param[in] guess: initial guess
  //! \param[in] residuals: initial residuals
  // AJ TODO Update comments above
  Diagnostic operator()(t_Vector &out, t_Vector const &x_guess, t_Vector &u_guess, t_Vector &v_guess, t_Vector &x_bar_guess, t_Vector const &res_guess, Real l2ball_epsilon_guess);

};

template <class SCALAR>
void PrimalDual<SCALAR>::iteration_step(t_Vector &out, t_Vector &residual, t_Vector &u,
					     t_Vector &v, t_Vector &x_bar) const {

  t_Vector prev_sol = out;
  t_Vector prev_u = u;
  t_Vector prev_v = v;

  auto const l2ball_proximal = proximal::L2Ball<Scalar>(l2ball_epsilon());
  
  // v_t = v_t-1 + Phi*x_bar - l2ball_prox(v_t-1 + Phi*x_bar)
  t_Vector temp = v + (Phi() * x_bar).eval();
  t_Vector v_prox;
  v_prox = l2ball_proximal(0, temp-target()) + target();
  v = temp - v_prox;
  
  // u_t = u_t-1 + Psi_dagger * x_bar_t-1 - l1norm_prox(s_t-1 + Psi_dagger * x_bar_t-1)
  t_Vector temp2 = u + (Psi().adjoint() * x_bar);
  t_Vector u_prox;
  proximal::l1_norm(u_prox, kappa()/sigma1(), temp2);
  u = temp2 - u_prox;
  
  //x_t = positive orth projection(x_t-1 - tau * (sigma1 * Psi * u + sigma2 * Phi dagger * v)) 
  out = prev_sol - tau()*(Psi()*u*sigma1() + Phi().adjoint()*v*sigma2());
  if(positivity_constraint()){
    out = psi::positive_quadrant(out);
  }
  x_bar = 2*out - prev_sol;
  residual = (Phi() * out).eval() - target();
    
}

 template <class SCALAR>
   typename PrimalDual<SCALAR>::Diagnostic PrimalDual<SCALAR>::
   operator()(t_Vector &out, t_Vector const &x_guess, t_Vector &u_guess, t_Vector &v_guess, t_Vector &x_bar_guess, t_Vector const &res_guess, Real l2ball_epsilon_guess) {
   PSI_HIGH_LOG("Performing Primal Dual");
   sanity_check(x_guess, u_guess, v_guess, x_bar_guess, res_guess, l2ball_epsilon_guess);
   
   t_Vector residual = res_guess;
   
   // Check if there is a user provided convergence function
   bool const has_user_convergence = static_cast<bool>(is_converged());
   bool converged = false;
   
   out = x_guess;
   t_uint niters(0);
   
   Vector<Real> weights = Vector<Real>::Ones(1);
   
   // TODO: Remove l1_weights
   // Required because some compilers have problems deducing the type of
   // l1_proximal_weights. This needs to be fixed in the long run but 
   // is ok for now.
   Vector<Real> l1_weights;
   if (l1_proximal_weights().size() == 1 && (l1_proximal_weights()(0)) == 0) {
    l1_weights = Vector<Real>::Ones(1);
   } else {
     l1_weights = l1_proximal_weights();
   }
   
   std::pair<Real, Real> objectives{psi::l1_norm(Psi().adjoint() * out, l1_weights), 0};
   
   for(; niters < itermax(); ++niters) {
     PSI_LOW_LOG("    - Iteration {}/{}", niters, itermax());
     iteration_step(out, residual, u_guess, v_guess, x_bar_guess);
     PSI_LOW_LOG("      - Sum of residuals: {}", residual.array().abs().sum());
     
     objectives.second = objectives.first;
     objectives.first = psi::l1_norm(Psi().adjoint() * out, l1_weights);
     t_real const relative_objective
       = std::abs(objectives.first - objectives.second) / objectives.first;
     PSI_LOW_LOG("    - objective: obj value = {}, rel obj = {}", objectives.first,
                 relative_objective);
     
    
     auto const residual_norm = psi::l2_norm(residual, weights);
     PSI_LOW_LOG("      - residual norm = {}, residual convergence = {}", residual_norm, residual_convergence());
     
     auto const user = (not has_user_convergence) or is_converged(out);
     auto const res = residual_convergence() <= 0e0 or residual_norm < residual_convergence();
     auto const rel = relative_variation() <= 0e0 or relative_objective < relative_variation();
     
     converged = user and res and rel;
     if(converged) {
       PSI_MEDIUM_LOG("    - converged in {} of {} iterations", niters, itermax());
       break;
     }
   }
   // check function exists, otherwise, don't know if convergence is meaningful
   if(not converged)
     PSI_ERROR("    - did not converge within {} iterations", itermax());
   
   return {niters, converged, std::move(residual)};
 }
 }
} /* psi::algorithm */
#endif
