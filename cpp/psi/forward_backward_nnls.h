#ifndef PSI_FORWARD_BACKWARD_NNLS_H
#define PSI_FORWARD_BACKWARD_NNLS_H

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
#include <cmath>

namespace psi {
namespace algorithm {

template <class SCALAR>
class ForwardBackward_nnls {
public:
	//! Scalar type
	typedef SCALAR value_type;
	//! Scalar type
	typedef value_type Scalar;
	//! Real type
	typedef typename real_type<Scalar>::type Real;
	//! Type of then underlying vectors
	typedef Vector <Scalar> t_Vector;
	//! Type of the Ψ and Ψ^H operations, as well as Φ and Φ^H
	typedef LinearTransform <t_Vector> t_LinearTransform;
	//! Type of the convergence function
	typedef ConvergenceFunction <Scalar> t_IsConverged;
	//! Type of the convergence function
	typedef ProximalFunction <Scalar> t_Proximal;
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
	template<class DERIVED>
	ForwardBackward_nnls(Eigen::MatrixBase<DERIVED> const &target)
	: itermax_(std::numeric_limits<t_uint>::max()), is_converged_(),
	  relative_variation_(1e-8),
	  mu_(1),
	  Phi_(std::make_shared<t_LinearTransform>(linear_transform_identity<Scalar>())),
	  FISTA_(false),
	  target_(target) {}
	virtual ~ForwardBackward_nnls() {}

	// Macro helps define properties that can be initialized as in
	// auto pd  = ForwardBackward<float>().prop0(value).prop1(value);
#define PSI_MACRO(NAME, TYPE)                                                                     \
		TYPE const &NAME() const { return NAME##_; }                                                     \
		ForwardBackward_nnls<SCALAR> &NAME(TYPE const &NAME) {                                                \
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
	//! A function verifying convergence
	PSI_MACRO(is_converged, t_IsConverged);
	//!  Convergence of the relative variation of the objective functions
	//!  If negative, this convergence criteria is disabled.
	PSI_MACRO(relative_variation, Real);
	//! Gradient step
	PSI_MACRO(mu, Real);
	//! Measurement operator
	PSI_MACRO(Phi, std::shared_ptr<const t_LinearTransform>);

	//! FISTA
	PSI_MACRO(FISTA, bool);

#undef PSI_MACRO


//	//! Set Φ and Φ^† using arguments that psi::linear_transform understands
//	template<class... ARGS>
//	typename std::enable_if<sizeof...(ARGS) >= 1, ForwardBackward_nnls &>::type Phi(ARGS &&... args) {
//		Phi_ = std::make_shared<t_LinearTransform>(linear_transform(std::forward<ARGS>(args)...));
//		return *this;
//	}

	//! Vector of target measurements
	t_Vector const &target() const { return target_; }

	//! Sets the vector of target measurements
	template<class DERIVED>
	ForwardBackward_nnls<DERIVED> &target(Eigen::MatrixBase<DERIVED> const &target) {
		target_ = target;
		return *this;
	}

	//! Facilitates call to user-provided convergence function
	bool is_converged(t_Vector const &x) const {
		return static_cast<bool>(is_converged()) and is_converged()(x);
	}

	//! \brief Calls Primal Dual
	//! \param[in] guess: initial guess
	DiagnosticAndResult operator()() const {
		DiagnosticAndResult result;
		auto const guess = initial_guess();
		static_cast<Diagnostic &>(result) = operator()(result.x, std::get<0>(guess), std::get<1>(guess));
		return result;
	}

	//! Makes it simple to chain different calls to Primal Dual
	DiagnosticAndResult operator()(DiagnosticAndResult const &warmstart) const {
		DiagnosticAndResult result = warmstart;
		static_cast<Diagnostic &>(result) = operator()(result.x, warmstart.x, warmstart.residual);
		return result;
	}

	//! \brief Computes initial guess for x and the residual
	//! \details with y the vector of measurements
	//! - x = P(y)
	//! - residuals = x - y
	std::tuple <t_Vector, t_Vector> initial_guess() const {
		std::tuple <t_Vector, t_Vector> guess;
		std::get<0>(guess) = (Phi()->adjoint())*t_Vector::Zero(target().size());
		std::get<1>(guess) = -target();
		return guess;
	}

	protected:

	void iteration_step(t_Vector &out, t_Vector &residual) const;

	void iteration_step(t_Vector &out, t_Vector &residual, t_Vector &prev_sol, Real &t) const;

	//! Checks input makes sense
	void sanity_check(t_Vector const &x_guess, t_Vector const &res_guess) const {
		if ((target()).size() != ((*Phi())*x_guess).size())
			PSI_THROW("target and input vector have inconsistent sizes");
		if (target().size() != res_guess.size())
			PSI_THROW("target and residual vector have inconsistent sizes");
		if (not static_cast<bool>(is_converged()))
			PSI_WARN("No convergence function was provided: algorithm will run for {} steps", itermax());
	}

	//! \brief Calls Forward-Backward
	//! \param[out] out: Output vector x
	//! \param[in] guess: initial guess
	//! \param[in] residuals: initial residuals
	Diagnostic operator()(t_Vector &out, t_Vector const &guess, t_Vector const &res) const;

	//! Vector of measurements
	t_Vector target_;
};

//template<class SCALAR>
//void ForwardBackward_nnls<SCALAR>::iteration_step(t_Vector &out, t_Vector &residual) const {
//
//    t_Vector v1, v2;
//
//    v1 = Phi().adjoint() * residual;
//    out = out - mu() * v1;
//
//    out = psi::positive_quadrant(out);
//
//    v2 = Phi() * out;
//    residual = v2 - target();
//}

template<class SCALAR> // beware copies here, use a more general template
void ForwardBackward_nnls<SCALAR>::iteration_step(t_Vector &out, t_Vector &residual, t_Vector &tmp_out, Real &t) const {

	t_Vector v1, v2;
	t_Vector prev_out = out;

	v1 = Phi()->adjoint() * residual;
	out = tmp_out - mu() * v1;

	out = psi::positive_quadrant(out);

	if(FISTA()) {

		Real told = t; // keep t and t_old as reals, cast as complex numbers when necessary

		t = (1. + sqrt(1. + 4. * told * told)) / 2.;

		tmp_out = out + (told - 1.) / t * (out - prev_out);
	}
	else{

		tmp_out = out;
	}

	v2 = (*Phi()) * tmp_out;
	residual = v2 - target();
}

template<class SCALAR>
typename ForwardBackward_nnls<SCALAR>::Diagnostic ForwardBackward_nnls<SCALAR>::
operator()(t_Vector &out, t_Vector const &x_guess, t_Vector const &res_guess) const {
	PSI_LOW_LOG("Performing Forward Backward");
	sanity_check(x_guess, res_guess);

	t_Vector residual = res_guess;

	// Check if there is a user provided convergence function
	bool const has_user_convergence = static_cast<bool>(is_converged());
	bool converged = false;

	out = x_guess;
	t_uint niters(0);

	std::pair <Real, Real> objectives{psi::l2_norm(out), 0};

	Real t=1.;
	t_Vector prev_sol = out;

	for (niters = 0; niters < itermax(); ++niters) {
		PSI_LOW_LOG("    - Iteration {}/{}", niters, itermax());
		iteration_step(out, residual, prev_sol, t);
		PSI_LOW_LOG("      - Sum of residuals: {}", residual.array().abs().sum()); // to be modified ?

		objectives.second = objectives.first;
		objectives.first = psi::l2_norm(residual);
		Real const relative_objective
		= std::abs(objectives.first - objectives.second) / objectives.first;
		PSI_LOW_LOG("    - objective: obj value = {}, rel obj = {}", objectives.first,
				relative_objective);
		PSI_LOW_LOG("Iteration: {}\tresidual norm: {}\trel_obj: {}", niters, objectives.first,relative_objective);

		auto const user = (not has_user_convergence) or is_converged(out);
		// auto const res = residual_convergence() <= 0e0 or residual_norm < residual_convergence();
		auto const rel = relative_variation() <= 0e0 or relative_objective < relative_variation();

		converged = user and rel; //and res
		if (converged) {
			PSI_MEDIUM_LOG("    - converged in {} of {} iterations", niters, itermax());
			break;
		}
	}
	// check function exists, otherwise, don't know if convergence is meaningful
	if (not converged)
		PSI_ERROR("    - did not converge within {} iterations", itermax());

	return {niters, converged, std::move(residual)};
}
}
} /* psi::algorithm */

#endif //PSI_FORWARD_BACKWARD_NNLS_H
