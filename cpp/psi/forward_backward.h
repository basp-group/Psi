#ifndef FORWARD_BACKWARD_H
#define FORWARD_BACKWARD_H

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
#include "psi/mpi/decomposition.h"

namespace psi {
namespace algorithm {

//! \brief Primal Dual method
//! \details This is a basic implementation of the primal dual method using a forward backwards algorithm.
//! \f$\min_{x, y, z} f(x) + l(y) + h(z)\f$ subject to \f$Φx = y\f$, \f$Ψ^Hx = z\f$
//!  We are not implementing blocking or parallelism here.
template <class SCALAR> class ForwardBackward {
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
	};

	//! Setups ForwardBackward
	template <class DERIVED>
	ForwardBackward(Eigen::MatrixBase<DERIVED> const &target, Eigen::MatrixBase<DERIVED> const &l2_ball_center)
	: itermax_(std::numeric_limits<t_uint>::max()), is_converged_(),
	  l2ball_epsilon_(1), Ui_(Vector<Real>::Ones(target.size())),
	  relative_variation_(1e-8),
	  target_(target), l2_ball_center_(l2_ball_center),
	  decomp_(psi::mpi::Decomposition(false)) {} // l2_ball_center denotes the center of the l2-ball
	virtual ~ForwardBackward() {}

	// Macro helps define properties that can be initialized as in
	// auto pd  = ForwardBackward<float>().prop0(value).prop1(value);
#define PSI_MACRO(NAME, TYPE)                                                                     \
		TYPE const &NAME() const { return NAME##_; }                                                     \
		ForwardBackward<SCALAR> &NAME(TYPE const &NAME) {                                                \
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
	//! l2 ball radius
	PSI_MACRO(l2ball_epsilon, Real);
	//! A function verifying convergence
	PSI_MACRO(is_converged, t_IsConverged);
	//! Preconditioning matrix (diagonal, matrix, represented by a vector)
	PSI_MACRO(Ui, Vector<Real>);
	//!  Convergence of the relative variation of the objective functions
	//!  If negative, this convergence criteria is disabled.
	PSI_MACRO(relative_variation, Real);
	//! Decomposition object
	PSI_MACRO(decomp, psi::mpi::Decomposition);

#undef PSI_MACRO

	//! Vector of target measurements
	t_Vector const &target() const { return target_; }
	//! Sets the vector of target measurements
	template <class DERIVED> ForwardBackward<DERIVED> &target(Eigen::MatrixBase<DERIVED> const &target) {
		target_ = target;
		return *this;
	}

	//! Vector of l2_ball_center measurements
	t_Vector const &l2_ball_center() const { return l2_ball_center_; }
	//! Sets the vector of target measurements
	template <class DERIVED> ForwardBackward<DERIVED> &l2_ball_center(Eigen::MatrixBase<DERIVED> const &l2_ball_center) {
		l2_ball_center_ = l2_ball_center;
		return *this;
	}

	//! Facilitates call to user-provided convergence function
	bool is_converged(t_Vector const &x) const {
		return static_cast<bool>(is_converged()) and is_converged()(x);
	}

	//! \brief Calls Forward Backward (to be modified)
	//! \param[out] out: Output vector x
	Diagnostic operator()(t_Vector &out) const { return operator()(out, initial_guess()); }
	//! \brief Calls Forward Backward
	//! \param[out] out: Output vector x
	//! \param[in] guess: initial guess
	Diagnostic operator()(t_Vector &out, std::tuple<t_Vector, t_Vector> const &guess) const {
		return operator()(out, std::get<0>(guess), std::get<1>(guess));
	}
	//! \brief Calls Primal Dual
	//! \param[in] guess: initial guess
	DiagnosticAndResult operator()(std::tuple<t_Vector, t_Vector> const &guess) const {
		DiagnosticAndResult result;
		static_cast<Diagnostic &>(result) = operator()(result.x, guess);
		return result;
	}
	//! \brief Calls Primal Dual
	//! \param[in] guess: initial guess
	DiagnosticAndResult operator()() const {
		DiagnosticAndResult result;
		static_cast<Diagnostic &>(result) = operator()(result.x, initial_guess());
		return result;
	}
	//! Makes it simple to chain different calls to Primal Dual
	DiagnosticAndResult operator()(DiagnosticAndResult const &warmstart) const {
		DiagnosticAndResult result = warmstart;
		static_cast<Diagnostic &>(result) = operator()(result.x, warmstart.x, warmstart.residual);
		return result;
	}

	//! \brief Computes initial guess for x and the residual (to be modified)
	//! \brief Computes initial guess for x and the residual
	//! \details with y the vector of measurements
	//! - x = P(y)
	//! - residuals = x - y
	std::tuple<t_Vector, t_Vector> initial_guess() const {
		std::tuple<t_Vector, t_Vector> guess;
		proximal::L2Ball<Scalar> l2ball_proximal = proximal::L2Ball<Scalar>(l2ball_epsilon());
		std::get<0>(guess) = t_Vector::Zero(target().size()); //l2ball_proximal(target() - l2_ball_center()) + l2_ball_center();
		std::get<1>(guess) = std::get<0>(guess) - target();
		return guess;
	}

	protected:

	void iteration_step(t_Vector &out, t_Vector &residual, Real &mu) const;

	//! Checks input makes sense
	void sanity_check(t_Vector const &x_guess, t_Vector const &res_guess) const {
		if((Ui().array() <= 0.).any())
			PSI_THROW("inconsistent values in preconditioning vector (all entries must be positive)");
		if((target()).size() != x_guess.size())
			PSI_THROW("target and input vector have inconsistent sizes");
		if(target().size() != res_guess.size())
			PSI_THROW("target and residual vector have inconsistent sizes");
		if(not static_cast<bool>(is_converged()))
			PSI_WARN("No convergence function was provided: FB algorithm will run for {} steps", itermax());
	}

	//! \brief Calls Forward-Backward
	//! \param[out] out: Output vector x
	//! \param[in] guess: initial guess
	//! \param[in] residuals: initial residuals
	Diagnostic operator()(t_Vector &out, t_Vector const &guess, t_Vector const &res) const;

	//! Vector of measurements
	t_Vector target_;
	//! Center of the l2-ball
	t_Vector l2_ball_center_;
};

template <class SCALAR>
void ForwardBackward<SCALAR>::iteration_step(t_Vector &out, t_Vector &residual, Real &mu) const {

	proximal::L2Ball<Scalar> l2ball_proximal = proximal::L2Ball<Scalar>(l2ball_epsilon());

	// v_t = l2ball_prox(v_t-1 - mu*Ui*(v_t-1 - z))
	t_Vector temp = out - mu*(Ui().array()*(residual).array()).matrix(); // prev_sol - target()
	out = l2ball_proximal(temp - l2_ball_center()) + l2_ball_center();
	residual= out - target();
}

template <class SCALAR>
typename ForwardBackward<SCALAR>::Diagnostic ForwardBackward<SCALAR>::
operator()(t_Vector &out, t_Vector const &x_guess, t_Vector const &res_guess) const {
	if(decomp().global_comm().is_root()){
		PSI_LOW_LOG("Performing Forward Backward");
	}
	sanity_check(x_guess, res_guess);

	proximal::L2Ball<Scalar> l2ball_proximal = proximal::L2Ball<Scalar>(l2ball_epsilon());

	t_Vector residual = res_guess;

	// Check if there is a user provided convergence function
	bool const has_user_convergence = static_cast<bool>(is_converged());
	bool converged = false;

	out = x_guess;
	t_uint niters(0);
	Real mu = 1/(std::pow(Ui().maxCoeff(),2));

	for(niters = 0; niters < itermax(); ++niters) {
		t_Vector x_old = out;
		PSI_LOW_LOG("    - FB Iteration {}/{}", niters, itermax());
		iteration_step(out, residual, mu);
		// PSI_LOW_LOG("      - FB Sum of residuals: {}", residual.array().abs().sum()); // to be modified ?

		Real const relative_objective
		= (out - x_old).stableNorm() / out.stableNorm();
		PSI_LOW_LOG("    - FB objective: rel obj = {}", relative_objective);

		auto const user = (not has_user_convergence) or is_converged(out);
		auto const rel = relative_variation() <= 0e0 or relative_objective < relative_variation();

		converged = user and rel;
		if(converged) {
			PSI_MEDIUM_LOG("    - FB converged in {} of {} iterations", niters, itermax());
			break;
		}
	}
	// check function exists, otherwise, don't know if convergence is meaningful
	if(not converged)
		PSI_ERROR("    - FB did not converge within {} iterations", itermax());

	return {niters, converged, std::move(residual)};
}
}
} /* psi::algorithm */
#endif
