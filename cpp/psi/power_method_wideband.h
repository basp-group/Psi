#ifndef PSI_POWER_METHOD_WIDEBAND_H
#define PSI_POWER_METHOD_WIDEBAND_H

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
t_real power_method_wideband(const std::vector<psi::LinearTransform<T>>  &op, const t_uint &niters, const t_real &relative_difference, const T2 &initial_matrix) {
	/* power method, returns the sqrt of the largest eigen value of a linear operator composed
     with its adjoint niters:: max number of iterations 
     relative_difference:: percentage difference at which eigen value has converged
	 */
	if(niters <= 0)
		return 1;
	t_real estimate_eigen_value = 1;
	t_real old_value = 0;
	T2 estimate_eigen_matrix = initial_matrix; // make sure this is a matrix...
	estimate_eigen_matrix = estimate_eigen_matrix / estimate_eigen_matrix.matrix().norm();
	PSI_DEBUG("Starting power method");
	PSI_DEBUG(" -[PM] Iteration: 0, norm = {}", estimate_eigen_value);
	for(t_int i = 0; i < niters; ++i) {
		for(t_int l = 0; l < op.size(); ++l){
			estimate_eigen_matrix.col(l) = (op[l].adjoint() * (op[l] * estimate_eigen_matrix.col(l)).eval()).eval();
		}
		estimate_eigen_value = estimate_eigen_matrix.matrix().norm();
		PSI_DEBUG("Iteration: {}, norm = {}", i + 1, estimate_eigen_value);
		if(estimate_eigen_value <= 0)
			throw std::runtime_error("Error in operator.");
		if(estimate_eigen_value != estimate_eigen_value)
			throw std::runtime_error("Error in operator or data corrupted.");
		estimate_eigen_matrix = estimate_eigen_matrix / estimate_eigen_value;
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
template <class SCALAR> class PowerMethodWideband {
public:
	//! Scalar type
	typedef SCALAR value_type;
	//! Scalar type
	typedef value_type Scalar;
	//! Real type
	typedef typename real_type<Scalar>::type Real;
	//! Type of then underlying vectors
	typedef Vector<Scalar> t_Vector;
	//! Type of then underlying matrices
	typedef Matrix<Scalar> t_Matrix;
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
		Matrix<Scalar> eigenvector;
	};

	//! Setups ProximalADMM
	PowerMethodWideband() : itermax_(std::numeric_limits<t_uint>::max()), tolerance_(1e-8), decomp_(psi::mpi::Decomposition(false)) {}
	virtual ~PowerMethodWideband() {}

	// Macro helps define properties that can be initialized as in
	// auto sdmm  = ProximalADMM<float>().prop0(value).prop1(value);
#define PSI_MACRO(NAME, TYPE)                                                                      \
		TYPE const &NAME() const { return NAME##_; }                                                     \
		PowerMethodWideband<SCALAR> &NAME(TYPE const &NAME) {                                            \
			NAME##_ = NAME;                                                                                \
			return *this;                                                                                  \
		}                                                                                                \
		\
protected:                                                                                         \
TYPE NAME##_;                                                                                    \
\
public:

	//! Maximum number of iterations // needs to be adapted from this point
	PSI_MACRO(itermax, t_uint);
	//! Convergence criteria
	PSI_MACRO(tolerance, Real);
	//! Decomposition object
	PSI_MACRO(decomp, psi::mpi::Decomposition);
#undef PSI_MACRO
	//! \brief Calls the power method for A.adjoint() * A
	DiagnosticAndResult AtA(std::vector<std::vector<std::shared_ptr<t_LinearTransform>>> &A, t_Matrix const &input) const;

	//! \brief Calls the power method for A a std::vector, where A[l] are matrices (l < A.size())
	template <class DERIVED>
	DiagnosticAndResult operator()(std::vector<Eigen::DenseBase<DERIVED>> const &A, t_Matrix const &input) const;

	//! \brief Calls the power method for a given matrix-vector multiplication function
	DiagnosticAndResult operator()(OperatorFunction<t_Matrix> const &op, t_Matrix const &input) const;

protected:
};

template <class SCALAR>
typename PowerMethodWideband<SCALAR>::DiagnosticAndResult
PowerMethodWideband<SCALAR>::AtA(std::vector<std::vector<std::shared_ptr<t_LinearTransform>>> &A, t_Matrix const &input) const {
	auto const op = [&A, this](t_Matrix &out, t_Matrix const &input) -> void {
		for(t_uint f = 0; f < A.size(); ++f){
			out.col(f).setZero();
			// Need to parallelise as A is parallelised
			for(t_uint b = 0; b < A[f].size(); ++b){
				out.col(f) += (A[f][b]->adjoint() * ((*A[f][b]) * input.col(f)).eval()).eval();
			}
			// COMMUNICATION HERE TO REDUCE THE local out's
			if(decomp().parallel_mpi() and decomp().my_frequencies()[f].freq_comm.size()>1){
				t_Vector temp_col = out.col(f);
				decomp().my_frequencies()[f].freq_comm.all_sum_all(temp_col);
				out.col(f) = temp_col;
			}
		}
	};
	return operator()(op, input);
}

template <class SCALAR>
template <class DERIVED>
typename PowerMethodWideband<SCALAR>::DiagnosticAndResult PowerMethodWideband<SCALAR>::
operator()(std::vector<Eigen::DenseBase<DERIVED>> const &A, t_Matrix const &input) const {
	auto const op = [&A](t_Matrix &out, t_Matrix const &input, int const rank) -> void {
		for(int l = 0; l < A.size(); ++l){
			Matrix<Scalar> const Ad = A[l].derived();
			out.col(l) = Ad * input.col(l);
		}
	};
	return operator()(op, input);
}

template <class SCALAR>
typename PowerMethodWideband<SCALAR>::DiagnosticAndResult PowerMethodWideband<SCALAR>::
operator()(OperatorFunction<t_Matrix> const &op, t_Matrix const &input) const {
	PSI_LOW_LOG("Computing the upper bound of a given operator");
	t_Matrix eigenvector = input.normalized();
	PSI_LOW_LOG("    - eigenvector norm {}", eigenvector.norm());
	typename t_Vector::Scalar previous_magnitude = 1;
	bool converged = false;
	t_uint niters = 0;

	for(; niters < itermax() and converged == false; ++niters) {
		auto eigenvector_out = eigenvector;
		op(eigenvector_out, eigenvector);
		eigenvector = eigenvector_out;
		typename t_Vector::Scalar magnitude = eigenvector.squaredNorm();
		auto real_comp = magnitude.real();
		auto imag_comp = magnitude.imag();
		decomp().global_comm().distributed_sum(&real_comp,decomp().global_comm().root_id());
		real_comp = decomp().global_comm().broadcast(real_comp, decomp().global_comm().root_id());
		decomp().global_comm().distributed_sum(&imag_comp,decomp().global_comm().root_id());
		imag_comp = decomp().global_comm().broadcast(imag_comp, decomp().global_comm().root_id());
		magnitude.real(real_comp);
		magnitude.imag(imag_comp);
		magnitude = sqrt(magnitude);
		auto const rel_val = std::abs((magnitude - previous_magnitude) / previous_magnitude);
		converged = rel_val < tolerance();

		if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
			PSI_HIGH_LOG("    - Power Method Wideband Iteration {}/{} -- norm: {}", niters, itermax(), magnitude);
		}

		eigenvector /= magnitude;
		previous_magnitude = magnitude;
	}

	// check function exists, otherwise, don't know if convergence is meaningful
	if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
		if(not converged) {
			PSI_WARN("    - Power Method Wideband did not converge within {} iterations", itermax());
		} else {
			PSI_INFO("    - Power Method Wideband converged in {} of {} iterations", niters, itermax());
		}
	}
	return DiagnosticAndResult{itermax(), converged, previous_magnitude, eigenvector.normalized()};
}
}
} /* psi::algorithm */
#endif
