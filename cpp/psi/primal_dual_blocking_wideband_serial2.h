#ifndef PSI_PRIMAL_DUAL_BLOCKING_WIDEBAND2_H
#define PSI_PRIMAL_DUAL_BLOCKING_WIDEBAND2_H

#include "psi/config.h"
#include <iostream>
#include <iomanip>
#include <functional>
#include <limits>
#include "psi/exception.h"
#include "psi/linear_transform.h"
#include "psi/linear_transform_operations.h"
#include "psi/logging.h"
#include "psi/types.h"
#include "psi/proximal.h"
#include "psi/utilities.h"
#include "psi/forward_backward.h"

namespace psi {
namespace algorithm {

//! \brief Primal Dual method
//! \details This is a basic implementation of the primal dual method using a forward backwards algorithm.
//! \f$\min_{, y, z} f() + l(y) + h(z)\f$ subject to \f$Φx = y\f$, \f$Ψ^Hx = z\f$
//!  We are not implementing blocking or parallelism here.
template <class SCALAR> class SerialPrimalDualBlockingWideband2 {
public:
  //! Scalar type
  typedef SCALAR value_type;
  //! Scalar type
  typedef value_type Scalar;
  //! Real type
  typedef typename real_type<Scalar>::type Real;
  //! Type of the underlying vectors
  typedef Vector<Scalar> t_Vector;
  //! Type of the Ψ and Ψ^H operations, as well as Φ and Φ^H
  typedef LinearTransform<t_Vector> t_LinearTransform;
  //! Type of the convergence function
  typedef ConvergenceMatrixFunction<Scalar> t_IsConverged;
  //! Type of the convergence function
  typedef ProximalFunction<Scalar> t_Proximal;
  typedef psi::Matrix<Scalar> t_Matrix;
  //! Values indicating how the algorithm ran
  struct Diagnostic {
    //! Number of iterations
    t_uint niters;
    //! Wether convergence was achieved
    bool good;
    //! Residual from the last iteration
    std::vector<std::vector<t_Vector>> residual;

    Diagnostic(t_uint niters = 0u, bool good = false)
        : niters(niters), good(good), residual({std::vector<t_Vector>(1, t_Vector::Zero(1))}) {}
    Diagnostic(t_uint niters, bool good, std::vector<std::vector<t_Vector>> &&residual)
        : niters(niters), good(good), residual(std::move(residual)) {}
 };
  //! Holds result vector and dual variables as well
  struct DiagnosticAndResult : public Diagnostic {
    //! Output 
    t_Matrix x;
    //! Dual variables p, u and v
    t_Matrix p;
    t_Matrix u;
    t_Matrix x_bar; 
    std::vector<std::vector<t_Vector>> v;
    //! epsilon parameter (updated for real data)
    std::vector<Vector<Real>> epsilon;
  };

  //! Setups SerialPrimalDualBlockingWideband2
  template <class T>
    SerialPrimalDualBlockingWideband2(std::vector<std::vector<T>> const &target, const t_uint &image_size, const std::vector<Vector<Real>> &l2ball_epsilon,
                               std::vector<std::vector<std::shared_ptr<const t_LinearTransform>>> &Phi, std::vector<std::vector<Vector<Real>>> const &Ui, const t_Matrix &x0)
    : itermax_(std::numeric_limits<t_uint>::max()), is_converged_(),
      mu_(1.), tau_(1.), kappa1_(1.), kappa2_(1.), kappa3_(1.),
      levels_(1),
      l21_proximal_weights_(Vector<Real>::Ones(1)), nuclear_proximal_weights_(Vector<Real>::Ones(image_size)), // see how these elements can be initialized
      Phi_(Phi),
      Psi_(linear_transform_identity<Scalar>()),
      residual_convergence_(1.001), 
      relative_variation_(1e-4), relative_variation_x_(1e-4), positivity_constraint_(true), update_epsilon_(false), lambdas_(Vector<Real>::Ones(3)), P_(20), adaptive_epsilon_start_(200),
      target_(target), image_size_(image_size), l2ball_epsilon_(l2ball_epsilon),
      preconditioning_(false), Ui_(Ui), itermax_fb_(20), relative_variation_fb_(1e-8), x0_(x0) {}
  virtual ~SerialPrimalDualBlockingWideband2() {}


// Macro helps define properties that can be initialized as in
// auto pd  = SerialPrimalDualBlockingWideband2<float>().prop0(value).prop1(value);
#define PSI_MACRO(NAME, TYPE)                                                                      \
  TYPE const &NAME() const { return NAME##_; }                                                     \
  SerialPrimalDualBlockingWideband2<SCALAR> &NAME(TYPE const &NAME) {                                             \
    NAME##_ = NAME;                                                                                \
    return *this;                                                                                  \
  }                                                                                                \
                                                                                                   \
protected:                                                                                         \
  TYPE NAME##_;                                                                                    \
                                                                                                   \
public:

  // Add for monitroing purposes
  //! Ground truth image
  PSI_MACRO(x0, t_Matrix);
  //--

  //! Maximum number of iterations
  PSI_MACRO(itermax, t_uint);
  //! Image size
  PSI_MACRO(image_size, t_uint);
  //! μ
  PSI_MACRO(mu, Real);
  //! τ
  PSI_MACRO(tau, Real);
  //! κ1
  PSI_MACRO(kappa1, Real);
  //! κ2
  PSI_MACRO(kappa2, Real);
  //! κ3
  PSI_MACRO(kappa3, Real);
  //! Number of dictionaries used in the wavelet operator
  PSI_MACRO(levels, t_uint);
  //! Weights for the l21-norm regularization
  PSI_MACRO(l21_proximal_weights, Vector<Real>);
  //! Weights for the nuclear-norm regularization
  PSI_MACRO(nuclear_proximal_weights, Vector<Real>);
  //! l2 ball bound
  PSI_MACRO(l2ball_epsilon, std::vector<Vector<Real>>);
  //! Threshold for the update of l2ball_epsilon
  PSI_MACRO(lambdas, Vector<Real>);
  //! Threshold for the relative variation of the solution (triggers the update of l2ball_epsilon)
  PSI_MACRO(relative_variation_x, Real);
  //! Minimum number of iterations between two updates of the l2ball_epsilon
  PSI_MACRO(P, t_uint);
  //! Number of iterations after which l2ball_epsilon can be updated
  PSI_MACRO(adaptive_epsilon_start, t_uint);
  //! A function verifying convergence
  PSI_MACRO(is_converged, t_IsConverged);
  //! Measurement operator
  PSI_MACRO(Phi, std::vector<std::vector<std::shared_ptr<const t_LinearTransform>>>);
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
  //! Enforce whether the ball radii need to be updated or not
  PSI_MACRO(update_epsilon, bool);  
  //! Preconditioning variables
  PSI_MACRO(preconditioning, bool);
  //! Preconditioning vector (equivalent to a diagonal preconditioner)
  PSI_MACRO(Ui, std::vector<std::vector<Vector<Real>>>);
  //! Maximum number of inner iterations (projection onto the ellipsoid)
  PSI_MACRO(itermax_fb, t_uint);
  //! Relative variation (stopping criterion)
  PSI_MACRO(relative_variation_fb, Real);
  
#undef PSI_MACRO

  //! Vector of target measurements
  std::vector<std::vector<t_Vector>> const &target() const { return target_; }
  //! Sets the vector of target measurements
  template <class T> SerialPrimalDualBlockingWideband2<T> &target(std::vector<EigenCell<T>> const &target) { // check this part again
    target_ = target;
    return *this;
  }

  //! Facilitates call to user-provided convergence function
  bool is_converged(t_Matrix const &x) const {
    return static_cast<bool>(is_converged()) and is_converged()(x);
  }
  
  //! \brief Calls Primal Dual
  //! \param[out] out: Output vector x
  Diagnostic operator()(t_Matrix &out) { return operator()(out, initial_guess()); }
  //! \brief Calls Primal Dual
  //! \param[out] out: Output vector x
  //! \param[in] guess: initial guess
  Diagnostic operator()(t_Matrix &out, std::tuple<t_Matrix, t_Matrix, t_Matrix, std::vector<std::vector<t_Vector>>, t_Matrix, std::vector<std::vector<t_Vector>>, std::vector<Vector<Real>>> const &guess) {
    return operator()(out, std::get<0>(guess), std::get<1>(guess), std::get<2>(guess), std::get<3>(guess), std::get<4>(guess), std::get<5>(guess), std::get<6>(guess));
  }
  //! \brief Calls Primal Dual
  //! \param[in] guess: initial guess
  DiagnosticAndResult operator()(std::tuple<t_Matrix, t_Matrix, t_Matrix, std::vector<std::vector<t_Vector>>, t_Matrix, std::vector<std::vector<t_Vector>>, std::vector<Vector<Real>>> const &guess) {
    DiagnosticAndResult result;
  // Need to refactor the operator to allow this to happen without the explicit copy between guess and result below.
    result.p = guess.p;
    result.u = guess.u;
    result.v = guess.v;
    result.x_bar = guess.x_bar;
    result.epsilon = guess.epsilon;
    static_cast<Diagnostic &>(result) = operator()(result.x, guess.x, result.p, result.u, result.v, result.x_bar, guess.residual, result.epsilon);
    return result;
  }
  //! \brief Calls Primal Dual
  //! \param[in] guess: initial guess
  DiagnosticAndResult operator()() {
    DiagnosticAndResult result;
    std::tuple<t_Matrix, t_Matrix, t_Matrix, std::vector<std::vector<t_Vector>>, t_Matrix, std::vector<std::vector<t_Vector>>, std::vector<Vector<Real>>> guess = initial_guess();
    result.p = std::get<1>(guess);
    result.u = std::get<2>(guess);
    result.v = std::get<3>(guess);
    result.x_bar = std::get<4>(guess);
    result.epsilon = std::get<6>(guess); 
    static_cast<Diagnostic &>(result) = operator()(result.x, std::get<0>(guess), result.p, result.u, result.v, result.x_bar,  std::get<5>(guess), result.epsilon);
// AJ I'd ideally do the below as it involves less memory operations, but I need to investigate how you do this in C++ more thoroughly first.
//    t_Vector initial_x, initial_residual;
//   (initial_x, result.s, result.v, result.x_bar, initial_residual) = initial_guess();
//   static_cast<Diagnostic &>(result) = operator()(result.x, initial_x, result.s, result.v, result.x_bar, initial_residual);
    return result;
  }
  //! Makes it simple to chain different calls to Primal Dual
  DiagnosticAndResult operator()(DiagnosticAndResult const &warmstart) {
    DiagnosticAndResult result = warmstart;
// Need to refactor the operator to allow this to happen without the explicit copy between warmstart and result below.
    result.p = warmstart.p;
    result.u = warmstart.u;
    result.v = warmstart.v;
    result.x_bar = warmstart.x_bar;
    result.epsilon = warmstart.epsilon;
    static_cast<Diagnostic &>(result) = operator()(result.x, warmstart.x, result.p, result.u, result.v, result.x_bar, warmstart.residual, result.epsilon);
    return result;
  }
  //! \brief Analysis operator Ψ and Ψ^† 
  template <class... ARGS>
  typename std::enable_if<sizeof...(ARGS) >= 1, SerialPrimalDualBlockingWideband2 &>::type Psi(ARGS &&... args) {
    Psi_ = linear_transform(std::forward<ARGS>(args)...);
    return *this;
  }

  //! \brief Computes initial guess for x and the residual
  //! \details with y the vector of measurements
  //! - x = Φ^T y
  //! - residuals = Φ x - y
  std::tuple<t_Matrix, t_Matrix, t_Matrix, std::vector<std::vector<t_Vector>>, t_Matrix, std::vector<std::vector<t_Vector>>, std::vector<Vector<Real>>> initial_guess() const {
    std::tuple<t_Matrix, t_Matrix, t_Matrix, std::vector<std::vector<t_Vector>>, t_Matrix, std::vector<std::vector<t_Vector>>, std::vector<Vector<Real>>> guess; // x, p, u, v, residual, epsilon
    std::get<0>(guess) = t_Matrix::Zero(image_size(), target().size()); // x
    std::get<1>(guess) = t_Matrix::Zero(image_size(), target().size()); // p
    std::get<2>(guess) = std::numeric_limits<Scalar>::epsilon()*t_Matrix::Ones(image_size()*levels(), target().size()); // u
    std::get<3>(guess) = std::vector<std::vector<t_Vector>>(target().size());         // v
    std::get<4>(guess) = t_Matrix::Zero(image_size(), target().size());               // x_bar
    std::get<5>(guess) = target(); //std::vector<std::vector<t_Vector>>(target().size());         // residual
    std::get<6>(guess) = l2ball_epsilon_;                                             // epsilon
    for (int l = 0; l < target().size(); ++l){
        std::get<3>(guess)[l] = std::vector<t_Vector>(target()[l].size()); // see if this is fine
        std::get<5>(guess)[l].reserve(target()[l].size());
        for (int b = 0; b < target()[l].size(); ++b){
            std::get<3>(guess)[l][b] = t_Vector::Zero(target()[l][b].rows()); // idem
            // std::get<5>(guess)[l].emplace_back(target()[l][b]);
        }
    }
    // Make sure the initial x respects the constraints
    if(positivity_constraint()){
      std::get<0>(guess) = psi::positive_quadrant(std::get<0>(guess));
    }
    return guess;
  }


protected:
  //! Vector of measurements
  std::vector<std::vector<t_Vector>> target_;
  
  void iteration_step(t_Matrix &out, std::vector<std::vector<t_Vector>> &residual, t_Matrix &p, t_Matrix &u, std::vector<std::vector<t_Vector>> &v, t_Matrix &x_bar, Vector<Real> &w_l21, Vector<Real> &w_nuclear) const;

  //! Checks input makes sense
  void sanity_check(t_Matrix const &x_guess, t_Matrix const &p_guess, t_Matrix const &u_guess, std::vector<std::vector<t_Vector>> const &v_guess, t_Matrix const &x_bar_guess, std::vector<std::vector<t_Vector>> const &res_guess, std::vector<Vector<Real>> const &l2ball_epsilon_guess) const {
    if((Phi()[0][0]->adjoint() * target()[0][0]).size() != x_guess.rows())
      PSI_THROW("target, adjoint measurement operator and input vector have inconsistent sizes");
    if(Phi().size() != target().size())
      PSI_THROW("target and vector of measurement operators have inconsistent sizes");
    if(x_bar_guess.size() != x_guess.size())
      PSI_THROW("x_bar and x have inconsistent sizes");
    if(target().size() != res_guess.size()) // check inner sizes?
      PSI_THROW("target and residual vector have inconsistent sizes");
    if(x_guess.cols() != target().size())
      PSI_THROW("target, measurement operator and input vector have inconsistent sizes");
    if(p_guess.size() != x_guess.size())
      PSI_THROW("input vector and dual variable p have inconsistent sizes");
    if(u_guess.size() != x_guess.size()*levels())
      PSI_THROW("input vector, measurement operator and dual variable u have inconsistent sizes");
    if(v_guess.size() != target().size())
      PSI_THROW("target and dual variable v have inconsistent sizes");
    if(not static_cast<bool>(is_converged()))
      PSI_WARN("No convergence function was provided: algorithm will run for {} steps", itermax());
    if(l2ball_epsilon_guess.size() != target().size())
      PSI_THROW("target and l2ball_epsilon have inconsistent sizes");
  }

  //! \brief Calls Primal Dual
  //! \param[out] out: Output vector 
  //! \param[in] guess: initial guess
  //! \param[in] residuals: initial residuals
  Diagnostic operator()(t_Matrix &out, t_Matrix const &guess, t_Matrix &p_guess, t_Matrix &u_guess, std::vector<std::vector<t_Vector>> &v_guess, t_Matrix &x_bar_guess, std::vector<std::vector<t_Vector>> const &res_guess, std::vector<Vector<Real>> &l2ball_epsilon_guess);
};

template <class SCALAR>
  void SerialPrimalDualBlockingWideband2<SCALAR>::iteration_step(t_Matrix &out, std::vector<std::vector<t_Vector>> &residual, t_Matrix &p, t_Matrix &u, std::vector<std::vector<t_Vector>> &v, t_Matrix &x_bar, Vector<Real> &w_l21, Vector<Real> &w_nuclear) const {

  t_Matrix prev_sol = out;

  // p_t = p_t-1 + x_bar - prox_nuclear_norm(p_t-1 + x_bar)
  t_Matrix tmp = p + x_bar;
  t_Matrix p_prox(out.rows(), out.cols());
  proximal::nuclear_norm(p_prox, tmp, w_nuclear);
  p = tmp - p_prox; 

  // Update dual variable u (sparsity in the wavelet domain)
  //! u_t[l] = u_t-1[l] + Psi.adjoint()*x_bar[l] - prox_l21(u_t-1[l] + Psi.adjoint()*x_bar[l])
  t_Matrix temp1_u(u.rows(), u.cols());
  for (int l = 0; l < target().size(); ++l){
    temp1_u.col(l) = u.col(l) + static_cast<t_Vector>(Psi().adjoint() * x_bar.col(l));
  }

  t_Matrix u_prox(u.rows(), u.cols());
  proximal::l21_norm(u_prox, temp1_u, w_l21);
  u = temp1_u - u_prox;
  t_Matrix temp2_u(out.rows(), out.cols());
  for (int l = 0; l < u.cols(); ++l){
    temp2_u.col(l) = static_cast<t_Vector>(Psi() * u.col(l));
  }
  

  // TO BE CHANGED
  // Update dual variable v (data fitting term) [to be updated to include the operations exposed by Adrian]
  //! v_t = v_t-1 + Phi*x_bar - l2ball_prox(v_t-1 + Phi*x_bar)
  // (in the following lines, the data are assumed to be ordered per blocks 
  // per channel, i.e., all blocks for channel 1, then 2, ...)

  // error here
  std::vector<std::vector<t_Vector>> Git_v(target().size());
  {
    for (int l = 0; l < target().size(); ++l){ //frequency channels

      auto const image_bar = Image<psi::t_complex>::Map(x_bar.col(l).data(), Phi()[0][0]->imsizey(), Phi()[0][0]->imsizex());
      psi::Matrix<psi::t_complex> x_hat = Phi()[0][0]->FFT(image_bar);

      for (int b = 0; b < target()[l].size(); ++b){   // assume the data are order per blocks per channel
        t_Vector temp;
        t_Vector v_prox;
        Git_v[l] = std::vector<t_Vector>(target()[l].size());

        if(preconditioning()){
            temp = v[l][b] + (Phi()[l][b]->G_function(x_hat)).eval().cwiseProduct(static_cast<t_Vector>(Ui()[l][b]));
            algorithm::ForwardBackward<SCALAR> ellipsoid_prox = algorithm::ForwardBackward<SCALAR>(temp.cwiseQuotient(Ui()[l][b]).eval(), target()[l][b])
                            .Ui(Ui()[l][b])
                            .itermax(itermax_fb())
                            .l2ball_epsilon(l2ball_epsilon_[l](b))
                            .relative_variation(relative_variation_fb());
            v_prox = ellipsoid_prox().x;
            v[l][b] = temp - v_prox.cwiseProduct(Ui()[l][b]);
        }
        else{      
          temp = v[l][b] + (Phi()[l][b]->G_function(x_hat)).eval();
          auto l2ball_proximal = proximal::L2Ball<SCALAR>(l2ball_epsilon_[l](b));
          v_prox = (l2ball_proximal(0, temp - target()[l][b]) + target()[l][b]);
          v[l][b] = temp - v_prox; 
        }
        // see of the opration is properly done here...
        Git_v[l][b] = Phi()[l][b]->G_function_adjoint(v[l][b]);
      }
    }
  }

  t_Matrix temp2_v = t_Matrix::Zero(out.rows(), out.cols());
  {
    for (int l = 0; l < target().size(); ++l){ // cols
      for (int b = 0; b < target()[l].size(); ++b){   // assume the data are order per blocks per channel
        psi::Matrix<psi::t_complex> v1 = psi::Matrix<psi::t_complex>::Map(Git_v[l][b].data(), 
        Phi()[0][0]->oversample_factor()*Phi()[0][0]->imsizey(),
        Phi()[0][0]->oversample_factor()*Phi()[0][0]->imsizex());
        psi::Image<psi::t_complex> im_tmp = Phi()[0][0]->inverse_FFT(v1);
        t_Vector v_tmp = t_Vector::Map(im_tmp.data(), im_tmp.size(), 1);
        temp2_v.col(l) = temp2_v.col(l) + v_tmp;
      }
    }
  }

  // Update primal variable x
  //! x_t = positive orth projection(x_t-1 - tau * (sigma1 * Psi * s + sigma2 * Phi dagger * v))
  out = prev_sol - tau()*(kappa1()*p + temp2_u*kappa2() + temp2_v*kappa3());
  if(positivity_constraint()){
    out = psi::positive_quadrant(out);
  }
  x_bar = 2*out - prev_sol;


  // Update the residue (in the following lines, the data are assumed to be ordered per blocks 
  // per channel, i.e., all blocks for channel 1, then 2, ...)
  {
    for (int l = 0; l < target().size(); ++l){ // cols     
      auto const image_out = Image<psi::t_complex>::Map(out.col(l).data(), Phi()[0][0]->imsizey(), Phi()[0][0]->imsizex());
      psi::Matrix<psi::t_complex> out_hat = Phi()[0][0]->FFT(image_out);      
      for (int b = 0; b < target()[l].size(); ++b){ 
        residual[l][b] = (Phi()[l][b]->G_function(out_hat)).eval() - target()[l][b];
      }
    }
  }
}

template <class SCALAR>
typename SerialPrimalDualBlockingWideband2<SCALAR>::Diagnostic SerialPrimalDualBlockingWideband2<SCALAR>::
  operator()(t_Matrix &out, t_Matrix const &x_guess, t_Matrix &p_guess, t_Matrix &u_guess, std::vector<std::vector<t_Vector>> &v_guess, t_Matrix &x_bar_guess, std::vector<std::vector<t_Vector>> const &res_guess, std::vector<Vector<Real>> &l2ball_epsilon_guess) {
  PSI_HIGH_LOG("Performing wideband primal-dual");
  sanity_check(x_guess, p_guess, u_guess, v_guess, x_bar_guess, res_guess, l2ball_epsilon_guess);

  std::vector<std::vector<t_Vector>> residual = res_guess;
  
  // Check if there is a user provided convergence function
  bool const has_user_convergence = static_cast<bool>(is_converged());
  bool converged = false;
  
  out = x_guess;
  l2ball_epsilon_ = l2ball_epsilon_guess;
  t_uint niters(0);

  // Lambda function to compute the wavelet-related regularization term
  auto wavelet_regularization = [](t_LinearTransform psi, const t_Matrix &X, const Vector<Real> &l21_weights) {
    t_Matrix Y(l21_weights.size(), X.cols());
    for (int l = 0; l < X.cols(); ++l){
      Y.col(l) = static_cast<t_Vector>(psi.adjoint() * X.col(l));
    }
    return psi::l21_norm(Y, l21_weights);
  };

  std::pair<Real, Real> objectives;
  Real nuclear_reg = psi::nuclear_norm(out, nuclear_proximal_weights());
  Real l21_reg = wavelet_regularization(Psi(), out, l21_proximal_weights());

  objectives.first = psi::nuclear_norm(out, nuclear_proximal_weights()) + mu()*wavelet_regularization(Psi(), out, l21_proximal_weights());
  objectives.second = 0.;

  // to be verified !! (use reserve )
  std::vector<Vector<int>> counter(target().size());
  std::vector<Vector<Real>> residual_norm(target().size());
  t_real total_residual_norm = 0;
  for(int l = 0; l < target().size(); ++l){
    counter[l] = Vector<int>::Zero(target()[l].size());
    residual_norm[l] = Vector<t_real>::Zero(target()[l].size());
    for(int b = 0; b < target()[l].size(); ++b){
        residual_norm[l](b) = residual[l][b].stableNorm();
    }
    total_residual_norm += residual_norm[l].squaredNorm();
  }
  total_residual_norm = std::sqrt(total_residual_norm);
  Vector<Real> w_l21 = mu()*l21_proximal_weights()/kappa2();
  Vector<Real> w_nuclear = nuclear_proximal_weights()/kappa1();

  // Debug section
  auto compute_snr = [](const t_Matrix &x, const t_Matrix &x0) {
      Real snr = 20*log10(x0.norm()/(x - x0).norm());
      return snr;
  };
  std::cout << std::scientific;
  std::cout.precision(5);
  auto snr = compute_snr(out, x0());
  std::cout << std::setw(16) << std::left << "Iteration: " << "\t" << std::setw(16) << std::left << "SNR" << std::setw(16) << std::left << "|y - Phi(x)|" << std::setw(16) << std::left << "f(x)" << "\n";
  std::cout << "--------------------------------------------------------------------\n";
  std::cout << std::setw(16) << std::left << 0 << "\t" << std::setw(16) << std::left << snr << std::setw(16) << std::left << total_residual_norm << objectives.first << "\n";
  std::cout << "--------------------------------------------------------------------\n";
  //--
  
  for(; niters < itermax(); ++niters) {

    t_Matrix x_old = out;

    PSI_LOW_LOG("    - Iteration {}/{}", niters, itermax());
    iteration_step(out, residual, p_guess, u_guess, v_guess, x_bar_guess, w_l21, w_nuclear);

    total_residual_norm = 0;
    t_real l2ball_epsilon_norm = 0;
    for(int l = 0; l < target().size(); ++l){
        for(int b = 0; b < target()[l].size(); ++b){
            residual_norm[l](b) = residual[l][b].stableNorm();
        }
        total_residual_norm += residual_norm[l].squaredNorm();
        l2ball_epsilon_norm += l2ball_epsilon_[l].squaredNorm();
    }
    total_residual_norm = std::sqrt(total_residual_norm);
    l2ball_epsilon_norm = std::sqrt(l2ball_epsilon_norm);
    PSI_LOW_LOG("    - residual norm = {}, residual convergence = {}", total_residual_norm, residual_convergence() * (l2ball_epsilon_norm));

    // update objective function
    objectives.second = objectives.first;
    objectives.first = psi::nuclear_norm(out, nuclear_proximal_weights()) + mu()*wavelet_regularization(Psi(), out, l21_proximal_weights());
    Real const relative_objective
        = std::abs(objectives.first - objectives.second) / objectives.first;
    PSI_LOW_LOG("    - objective: obj value = {}, rel obj = {}", objectives.first,
                 relative_objective);
    
    auto const user = (not has_user_convergence) or is_converged(out);
    auto const res = (residual_convergence() <= 0e0) or
                            (total_residual_norm < residual_convergence() * (l2ball_epsilon_norm));
    auto const rel = relative_variation() <= 0e0 or relative_objective < relative_variation();

    converged = user and res and rel;
    if(converged) {
      PSI_MEDIUM_LOG("    - converged in {} of {} iterations", niters, itermax());
      break;
    }

    auto const rel_x = (out - x_old).norm();

    // to be changed here
    // update l2ball_epsilons
    if(update_epsilon() && niters > adaptive_epsilon_start() && 
       rel_x < relative_variation_x()*out.norm()){     
      for(int l = 0; l < residual_norm.size(); ++l){
          for(int b = 0; b < residual_norm[l].size(); ++b){
            bool eps_flag = residual_norm[l](b) < lambdas()(0) * l2ball_epsilon_[l](b) or
                            residual_norm[l](b) > lambdas()(1) * l2ball_epsilon_[l](b);
            if(niters > counter[l](b) + P() and eps_flag){
                std::cout << "Update epsilon for block " << b << ", channel " << l << " at iteration " << niters << std::endl;
                std::cout << "epsilon before: " << l2ball_epsilon_[l](b) << "\t" << "residual norm: "
                            << residual_norm[l](b) << "\t";
                l2ball_epsilon_[l](b) =
                        lambdas()(2) * residual_norm[l](b) + (1 - lambdas()(2)) * l2ball_epsilon_[l](b);
                std::cout << "epsilon after: " << l2ball_epsilon_[l](b) << std::endl;
                counter[l](b) = niters;
            }
          }
      }
    }
    // Add for monitoring purposes
    snr = compute_snr(out, x0());
    std::cout << std::setw(16) << std::left << niters+1 << "\t" << std::setw(16) << std::left << snr << std::setw(16) << std::left << total_residual_norm << objectives.first << "\n";
    std::cout << "--------------------------------------------------------------------\n";
    //--
  }

  // check function exists, otherwise, don't know if convergence is meaningful
  if(not converged)
    PSI_ERROR("    - did not converge within {} iterations", itermax());
  l2ball_epsilon_guess = l2ball_epsilon_;
  return {niters, converged, std::move(residual)};
} /* end of primal-dual */
} /* psi::algorithm */
} /* psi */
#endif
