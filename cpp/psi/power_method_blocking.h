#ifndef PSI_POWER_METHOD_BLOCKING_H
#define PSI_POWER_METHOD_BLOCKING_H

#include "psi/config.h"
#include <functional>
#include <limits>
#include <vector>
#include "psi/exception.h"
#include "psi/linear_transform.h"
#include "psi/logging.h"
#include "psi/types.h"
#include "psi/mpi/decomposition.h"


namespace psi {
namespace algorithm {

template <class T, class T2>
t_real power_method_blocking(const std::vector<psi::LinearTransform<T>> &op, const t_uint &niters, const t_real &relative_difference, const T2 &initial_matrix) {
	/* returns the sqrt of the largest eigen value of a linear operator composed with its adjoint
     niters:: max number of iterations relative_difference::percentage difference at which eigen
     value has converged */
	if(niters <= 0)
		return 1;
	t_real estimate_eigen_value = 1;
	t_real old_value = 0;
	T estimate_eigen_vector = initial_matrix;
	estimate_eigen_vector = estimate_eigen_vector / estimate_eigen_vector.matrix().norm();
	PSI_DEBUG("Starting power method");
	PSI_DEBUG(" -[PM] Iteration: 0, norm = {}", estimate_eigen_value);
	for(t_int i = 0; i < niters; ++i) {
		for(t_int l = 0; l < op.size(); ++l){
			estimate_eigen_vector = (op[l].adjoint() * (op[l] * estimate_eigen_vector).eval()).eval();
		}
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


//! \brief Eigenvalue and eigenvector for eigenvalue with largest magnitude
template <class SCALAR> class PowerMethodBlocking {
public:
	//! Scalar type
	typedef SCALAR value_type;
	//! Scalar type
	typedef value_type Scalar;
	//! Real type
	typedef typename real_type<Scalar>::type Real;
	//! Type of then underlying vectors
	typedef Vector<Scalar> t_Vector;
	//! Type of the Psu and Psi^H operations, as well as Phi and Phi^H
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

	//! Setups ProximalADMM
	PowerMethodBlocking() : itermax_(std::numeric_limits<t_uint>::max()), tolerance_(1e-8), decomp_(psi::mpi::Decomposition(false)) {}
	virtual ~PowerMethodBlocking() {}

	// Macro helps define properties that can be initialized as in
	// auto sdmm  = ProximalADMM<float>().prop0(value).prop1(value);
#define PSI_MACRO(NAME, TYPE)                                                                      \
		TYPE const &NAME() const { return NAME##_; }                                                     \
		PowerMethodBlocking<SCALAR> &NAME(TYPE const &NAME) {                                                    \
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
	//! Decomposition object
	PSI_MACRO(decomp, psi::mpi::Decomposition);
#undef PSI_MACRO
	//! \brief Calls the power method for A.adjoint() * A
	DiagnosticAndResult AtA(std::vector<std::shared_ptr<const t_LinearTransform>> &A, t_Vector const &input) const;

	//! \brief Calls the power method for A, with A a matrix
	template <class DERIVED>
	DiagnosticAndResult operator()(std::vector<std::shared_ptr<const Eigen::DenseBase<DERIVED>>> &A, t_Vector const &input) const;

	//! \brief Calls the power method for a given matrix-vector multiplication function
	DiagnosticAndResult operator()(OperatorFunction<t_Vector> const &op, t_Vector const &input) const;

protected:
};


template <class SCALAR>
typename PowerMethodBlocking<SCALAR>::DiagnosticAndResult
PowerMethodBlocking<SCALAR>::AtA(std::vector<std::shared_ptr<const t_LinearTransform>> &A, t_Vector const &input) const {
	auto const op = [&A, this](t_Vector &out, t_Vector const &input) -> void {
		// Only do this on the root process
		out = t_Vector::Zero(input.size());
		// Need to parallelise as A is parallelised
		for(t_uint b = 0; b < A.size(); ++b){
			out += (A[b]->adjoint() * ((*A[b]) * input).eval()).eval();
		}
		// COMMUNICATION HERE TO REDUCE THE local out's
		if(decomp().parallel_mpi()){
			decomp().my_frequencies()[0].freq_comm.all_sum_all(out);
		}
	};
	return operator()(op, input);
}

template <class SCALAR>
template <class DERIVED>
typename PowerMethodBlocking<SCALAR>::DiagnosticAndResult PowerMethodBlocking<SCALAR>::
operator()(std::vector<std::shared_ptr<const Eigen::DenseBase<DERIVED>>>  &A, t_Vector const &input) const {
	auto const op = [&A](std::vector<t_Vector> &out, t_Vector const &input) -> void {
		// Need to parallelise as A is parallelised
		for(int b = 0; b < A.size(); ++b){
			Matrix<Scalar> const Ad = A[b]->derived();
			out[b] = Ad * input;
		}
	};
	return operator()(op, input);
}

template <class SCALAR>
typename PowerMethodBlocking<SCALAR>::DiagnosticAndResult PowerMethodBlocking<SCALAR>::
operator()(OperatorFunction<t_Vector> const &op, t_Vector const &input) const {
	PSI_INFO("Computing the upper bound of a given operator");
	t_Vector eigenvector = input.normalized();
	PSI_INFO("    - eigenvector norm {}", eigenvector.stableNorm());
	typename t_Vector::Scalar previous_magnitude = 1;
	bool converged = false;
	t_uint niters = 0;

	for(; niters < itermax() and converged == false; ++niters) {
		auto eigenvector_out = eigenvector;
		op(eigenvector_out, eigenvector); // problem with this operation: only zeros...
		eigenvector = eigenvector_out;
		typename t_Vector::Scalar const magnitude
		= eigenvector.stableNorm() / static_cast<Real>(eigenvector.size());
		auto const rel_val = std::abs((magnitude - previous_magnitude) / previous_magnitude);
		converged = rel_val < tolerance();
		PSI_INFO("    - Iteration {}/{} -- norm: {}", niters, itermax(), magnitude);

		eigenvector /= magnitude;
		previous_magnitude = magnitude;
	}
	// check function exists, otherwise, don't know if convergence is meaningful
	if(not converged) {
		PSI_WARN("    - did not converge within {} iterations", itermax());
	} else {
		PSI_INFO("    - converged in {} of {} iterations", niters, itermax());
	}
	return DiagnosticAndResult{itermax(), converged, previous_magnitude, eigenvector.normalized()};
}
}
} /* psi::algorithm */
#endif
