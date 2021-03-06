#ifndef PSI_REWEIGHTED_WIDEBAND_H
#define PSI_REWEIGHTED_WIDEBAND_H

#include <ctime>

#include "psi/linear_transform.h"
#include "psi/types.h"
#include "psi/mpi/decomposition.h"
#include "psi/mpi/scalapack.h"

namespace psi {
namespace algorithm {
template <class ALGORITHM> class Reweighted;

//! Factory function to create an l0-approximation by reweighting an l1 norm
template <class ALGORITHM>
Reweighted<ALGORITHM>
reweighted(ALGORITHM const &algo, typename Reweighted<ALGORITHM>::t_ReweighteeMat const &reweighteeL21);

//! \brief L0-approximation algorithm, through reweighting
//! \details This algorithm approximates \f$min_x ||Ψ^Tx||_0 + f(x)\f$ by solving the set of
//! problems \f$j\f$, \f$min_x ||W_jΨ^Tx||_1 + f(x)\f$ where the *diagonal* matrix \f$W_j\f$ is set
//! using the results from \f$j-1\f$: \f$ δ_j W_j^{-1} = δ_j + ||W_{j-1}Ψ^T||_1\f$. \f$δ_j\f$
//! prevents division by zero. It is a series which converges to zero. By default,
//! \f$δ_{j+1}=0.1δ_j\f$.
//!
//! The algorithm proceeds needs three forms of input:
//! - the inner algorithm, e.g. ImagingProximalADMM
//! - a function returning Ψ^Tx given x
//! - a function to modify the inner algorithm with new weights
template <class ALGORITHM> class Reweighted {
public:
	//! Inner-loop algorithm
	typedef ALGORITHM Algorithm;
	//! Scalar type
	typedef typename Algorithm::Scalar Scalar;
	//! Real type
	typedef typename real_type<Scalar>::type Real;
	//! Weight vector type
	typedef Vector<Real> WeightVector;
	//! Type of then underlying vectors
	typedef typename Algorithm::t_Vector XVector;
	typedef typename psi::Matrix<Scalar> XMatrix;
	//! Type of the convergence function
	typedef ConvergenceMatrixFunction<Scalar> t_IsConverged;
	//! \brief Type of the function that is subject to reweighting
	//! \details E.g. \f$Ψ^Tx\f$. Note that l1-norm is not applied here.
	typedef std::function<XMatrix(Algorithm const &, XMatrix const &)> t_ReweighteeMat;
	typedef std::function<XVector(Algorithm const &, XMatrix const &)> t_ReweighteeVec;
	//! Function to update delta at each turn
	typedef std::function<Real(Real)> t_DeltaUpdate;

	//! Output from running reweighting scheme
	struct ReweightedResult {
		//! Number of iterations (outer loop)
		t_uint niters;

		//! Whether convergence was achieved
		bool good;

		//! Result from last inner loop
		typename Algorithm::DiagnosticAndResult algo;
		//! Default construction
		ReweightedResult() : niters(0), good(false), algo() {}
	};

	Reweighted(Algorithm &algo, t_ReweighteeMat const &reweighteeL21)
	: algo_(algo), reweighteeL21_(reweighteeL21), itermax_(std::numeric_limits<t_uint>::max()),
	  min_delta_(0e0), is_converged_(), update_delta_([](Real delta) { return 1e-1 * delta; }) {}

	//! Underlying "inner-loop" algorithm
	Algorithm &algorithm() { return algo_; }

	//! Sets the underlying "inner-loop" algorithm
	Reweighted<Algorithm> &algorithm(Algorithm &algo) {
		algo_ = algo;
		return *this;
	}

	//! Sets the underlying "inner-loop" algorithm
	Reweighted<Algorithm> &algorithm(Algorithm &&algo) {
		algo_ = std::move(algo);
		return *this;
	}

	//! Function that needs to be Wideband
	//! \details E.g. \f$Ψ^Tx\f$. Note that l1-norm is not applied here.
	Reweighted<Algorithm> &reweighteeL21(t_ReweighteeMat const &rw) {
		reweighteeL21_ = rw;
		return *this;
	}
	//! Function that needs to be Wideband
	t_ReweighteeMat const &reweighteeL21() const { return reweighteeL21_; }

	//! Forwards to the reweightee function
	XMatrix reweighteeL21(XMatrix const &x) { return reweighteeL21()(algorithm(), x); }

	//! Maximum number of Wideband iterations
	t_uint itermax() const { return itermax_; }
	Reweighted &itermax(t_uint i) {
		itermax_ = i;
		return *this;
	}

	//! Lower limit for delta
	Real min_delta() const { return min_delta_; }
	Reweighted &min_delta(Real min_delta) {
		min_delta_ = min_delta;
		return *this;
	}

	//! Checks convergence of the reweighting scheme
	t_IsConverged const &is_converged() const { return is_converged_; }
	Reweighted &is_converged(t_IsConverged const &convergence) {
		is_converged_ = convergence;
		return *this;
	}
	bool is_converged(XMatrix const &x) const { return is_converged() ? is_converged()(x) : false; }

	//! \brief Performs reweighting
	//! \details This overload will compute an initial result without initial weights set to one.
	template <class INPUT>
	typename std::enable_if<not(std::is_same<INPUT, typename Algorithm::DiagnosticAndResult>::value
			or std::is_same<INPUT, ReweightedResult>::value),
			ReweightedResult>::type
			operator()(INPUT const &input);
	//! \brief Performs reweighting
	//! \details This overload will compute an initial result without initial weights set to one.
	ReweightedResult operator()() ;
	//! Reweighted algorithm, from prior call to inner-algorithm
	ReweightedResult operator()(typename Algorithm::DiagnosticAndResult const &warm);
	//! Reweighted algorithm, from prior call to reweighting algorithm
	ReweightedResult operator()(ReweightedResult const &warm, bool no_warm = false);

	//! Updates delta
	Real update_delta(Real delta) const { return update_delta()(delta); }
	//! Updates delta
	t_DeltaUpdate const &update_delta() const { return update_delta_; }
	//! Updates delta
	Reweighted<Algorithm> update_delta(t_DeltaUpdate const &ud) {
		update_delta_ = ud;
		return *this;
	}

protected:
	//! Inner loop algorithm
	Algorithm algo_;
	//! \brief Function that is subject to reweighting
	//! \details E.g. \f$Ψ^Tx\f$. Note that l1-norm is not applied here.
	t_ReweighteeMat reweighteeL21_;
	//! Maximum number of Wideband iterations
	t_uint itermax_;
	//! \brief Lower limit for delta
	Real min_delta_;
	//! Checks convergence
	t_IsConverged is_converged_;
	//! Updates delta at each turn
	t_DeltaUpdate update_delta_;
};

template <class ALGORITHM>
template <class INPUT>
typename std::
enable_if<not(std::is_same<INPUT, typename ALGORITHM::DiagnosticAndResult>::value
		or std::is_same<INPUT, typename Reweighted<ALGORITHM>::ReweightedResult>::value),
		typename Reweighted<ALGORITHM>::ReweightedResult>::type
		Reweighted<ALGORITHM>::operator()(INPUT const &input) {
	Algorithm algo = algorithm();
	return operator()(algo(input));
}

template <class ALGORITHM>
typename Reweighted<ALGORITHM>::ReweightedResult Reweighted<ALGORITHM>::operator()() {
        ReweightedResult result = ReweightedResult();
	return operator()(result, true);
}

template <class ALGORITHM>
typename Reweighted<ALGORITHM>::ReweightedResult Reweighted<ALGORITHM>::
operator()(typename Algorithm::DiagnosticAndResult const &warm) {
	ReweightedResult result;
	result.algo = warm;
	return (result);
}

template <class ALGORITHM>
typename Reweighted<ALGORITHM>::ReweightedResult Reweighted<ALGORITHM>::
operator()(ReweightedResult const &warm, bool no_warm) {

	double temptime, time1, time2, time3, time4;
	time1 = 0;
	time2 = 0;
	time3 = 0;
	time4 = 0;

	if(!algorithm().algorithm().decomp().parallel_mpi() or algorithm().algorithm().decomp().global_comm().is_root()){
		PSI_HIGH_LOG("Starting wideband re-weighting scheme");
	}
	// Copies inner algorithm, so that operator() can be constant
	Algorithm algo(algorithm());
	ReweightedResult result;
        result = ReweightedResult(warm);

	Real delta;

	if(no_warm){
		delta = 1;
	}else{
		delta = warm.algo.delta;
	}

	if(algorithm().algorithm().decomp().parallel_mpi() and algorithm().algorithm().decomp().my_root_wavelet_comm().size() != 1 and algorithm().algorithm().decomp().my_number_of_root_wavelets() != 0){
		delta = algorithm().algorithm().decomp().my_root_wavelet_comm().broadcast(delta, algorithm().algorithm().decomp().my_root_wavelet_comm().root_id());
	}
	if(!algorithm().algorithm().decomp().parallel_mpi() or algorithm().algorithm().decomp().global_comm().is_root()){
		PSI_HIGH_LOG("-   Re-weighting iteration: {}", warm.algo.current_reweighting_iter);
		PSI_HIGH_LOG("  - delta: {}", delta);
	}

	for(result.niters = 0; result.niters <= itermax(); ++result.niters) {


#ifdef PSI_OPENMP
		temptime = omp_get_wtime();
#endif
		if(no_warm and result.niters == 0){
		  result.algo = algo();
		}else{
		  result.algo = algo(result.algo);
		}

		bool good_result = false;

		if(!algorithm().algorithm().decomp().parallel_mpi() or algorithm().algorithm().decomp().global_comm().is_root()){
			if(is_converged(result.algo.x)) {
				PSI_HIGH_LOG("Re-weighting scheme converged in {} iterations", result.niters);
				good_result = true;
			}
		}
		good_result = algorithm().algorithm().decomp().global_comm().broadcast(good_result, algorithm().algorithm().decomp().global_comm().root_id());

#ifdef PSI_OPENMP
		temptime = omp_get_wtime() - temptime;
		time1 += temptime;
#endif

		if(good_result){
			result.good = true;
			break;
		}
                
		//! Remove the adaptive_epsilon_start restriction after the first re-weighting iteration.
		//! This allows restarting from saved runs without forcing the restriction to be undertaken multiple times.
		if(result.niters >= 1){
			algo.algorithm().adaptive_epsilon_start(0);
		}
		if(!algorithm().algorithm().decomp().parallel_mpi() or algorithm().algorithm().decomp().global_comm().is_root()){
			PSI_HIGH_LOG("Re-weighting iteration {}/{} ", result.niters, itermax());
			PSI_HIGH_LOG("  - delta: {}", delta);
		}

		psi::Matrix<t_complex> partial;

#ifdef PSI_OPENMP
		temptime = omp_get_wtime();
#endif
		if(!algorithm().algorithm().decomp().parallel_mpi() or algorithm().algorithm().decomp().my_number_of_root_wavelets() != 0){
			psi::Matrix<t_complex> local_x;
			if(algorithm().algorithm().decomp().my_root_wavelet_comm().size() != 1){
				local_x = algorithm().algorithm().decomp().my_root_wavelet_comm().broadcast(result.algo.x, algorithm().algorithm().decomp().my_root_wavelet_comm().root_id());
				partial = reweighteeL21(local_x);
			}else{
				partial = reweighteeL21(result.algo.x);
			}
			result.algo.weightsL21 = delta / (delta + partial.rowwise().norm().array());
		}

#ifdef PSI_OPENMP
		temptime = omp_get_wtime() - temptime;
		time2 += temptime;
#endif

#ifdef PSI_OPENMP
		temptime = omp_get_wtime();
#endif

		if(!algorithm().algorithm().decomp().parallel_mpi() or algorithm().algorithm().decomp().global_comm().is_root()){
			PSI_HIGH_LOG("Norm of sigma in reweighting: {}", result.algo.sigma.stableNorm());
			result.algo.weightsNuclear = delta / (delta + result.algo.sigma.array().abs());
			result.algo.delta = delta;
			result.algo.current_reweighting_iter = result.niters;
		}


#ifdef PSI_OPENMP
		temptime = omp_get_wtime() - temptime;
		time3 += temptime;
#endif

#ifdef PSI_OPENMP
		temptime = omp_get_wtime();
#endif
		if(!algorithm().algorithm().decomp().parallel_mpi() or algorithm().algorithm().decomp().global_comm().is_root()){
			delta = std::max(min_delta(), update_delta(delta));
		}
		if(algorithm().algorithm().decomp().parallel_mpi() and algorithm().algorithm().decomp().my_root_wavelet_comm().size() != 1 and algorithm().algorithm().decomp().my_number_of_root_wavelets() != 0){
			delta = algorithm().algorithm().decomp().my_root_wavelet_comm().broadcast(delta, algorithm().algorithm().decomp().my_root_wavelet_comm().root_id());
		}

#ifdef PSI_OPENMP
		temptime = omp_get_wtime() - temptime;
		time4 += temptime;
#endif

	}
	// result is always good if no convergence function is defined
	if(not is_converged()){
		result.good = true;
	}else if(not result.good){
		if(!algorithm().algorithm().decomp().parallel_mpi() or algorithm().algorithm().decomp().global_comm().is_root()){
			PSI_ERROR("Re-weighting scheme did not converge in {} iterations", itermax());
		}
	}

	if(algorithm().algorithm().decomp().global_comm().is_root()){
		PSI_HIGH_LOG("{} ReweightTime: 1: {} 2: {} 3: {} 4: {}",
				algorithm().algorithm().decomp().global_comm().rank(),
				(float)time1, (float)time2, (float)time3, (float)time4);
	}

	return result;
}

//! Factory function to create an l0-approximation by reweighting an l1 norm
template <class ALGORITHM>
Reweighted<ALGORITHM>
reweighted(ALGORITHM const &algo, typename Reweighted<ALGORITHM>::t_ReweighteeMat const &reweighteeL21) {
	return {algo, reweighteeL21};
}

template <class SCALAR> class PrimalDualWideband;
template <class ALGORITHM> class PositiveQuadrant;
template <class T>
Eigen::CwiseUnaryOp<const details::ProjectPositiveQuadrant<typename T::Scalar>, const T>
positive_quadrant(Eigen::DenseBase<T> const &input);

template <class SCALAR> class PrimalDualWidebandBlocking;
template <class SCALAR>
Reweighted<PositiveQuadrant<PrimalDualWidebandBlocking<SCALAR>>>
reweighted(PrimalDualWidebandBlocking<SCALAR> &algo) {

	PositiveQuadrant<PrimalDualWidebandBlocking<SCALAR>> posq = positive_quadrant(algo);
	typedef typename std::remove_const<decltype(posq)>::type Algorithm;
	typedef Reweighted<Algorithm> RW;

	auto reweighteeL21
	= [](Algorithm const &posq, typename RW::XMatrix const &x) -> typename RW::XMatrix {
		typename RW::XMatrix Y(posq.algorithm().image_size()*posq.algorithm().decomp().my_number_of_root_wavelets(), x.cols());
		for (int l = 0; l < x.cols(); ++l){
			Y.col(l) = static_cast<typename RW::XVector>(posq.algorithm().Psi_Root().adjoint() * x.col(l));
		}
		return Y;

	};


	return {posq, reweighteeL21};
}


} // namespace algorithm
} // namespace psi
#endif
