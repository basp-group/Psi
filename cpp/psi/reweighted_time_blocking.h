#ifndef PSI_REWEIGHTED_TIME_BLOCKING_H
#define PSI_REWEIGHTED_TIME_BLOCKING_H

#include "psi/linear_transform.h"
#include "psi/types.h"
#include "psi/mpi/decomposition.h"

namespace psi {
namespace algorithm {
template <class ALGORITHM> class Reweighted;

//! Factory function to create an l0-approximation by reweighting an l1 norm
template <class ALGORITHM>
Reweighted<ALGORITHM>
reweighted(ALGORITHM const &algo,
		typename Reweighted<ALGORITHM>::t_SetDelta const &set_delta,
		typename Reweighted<ALGORITHM>::t_Reweightee const &reweightee);

//! \brief L0-approximation algorithm, through reweighting
//! \details This algorithm approximates \f$min_x ||Ψ^Tx||_0 + f(x)\f$ by solving the set of
//! problems \f$j\f$, \f$min_x ||W_jΨ^Tx||_1 + f(x)\f$ where the *diagonal* matrix \f$W_j\f$ is set
//! using the results from \f$j-1\f$: \f$ δ_j W_j^{-1} = δ_j + ||W_{j-1}Ψ^T||_1\f$. \f$δ_j\f$
//! prevents division by zero. It is a series which converges to zero. By default,
//! \f$δ_{j+1}=0.1δ_j\f$.
//!
//! The algorithm proceeds needs three forms of input:
//! - the inner algorithm, e.g. PrimalDualTimeBlocking
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
	//! Type of the convergence function
	typedef ConvergenceFunction<Scalar> t_IsConverged;
	//! \brief Type of the function that is subject to reweighting
	//! \details E.g. \f$Ψ^Tx\f$. Note that l1-norm is not applied here.
	typedef std::function<XVector(Algorithm const &, XVector const &)> t_Reweightee;
	//! Type of the function to set delta
	typedef std::function<void(Algorithm &, Real const)> t_SetDelta;
	//! Function to update delta at each turn
	typedef std::function<Real(Real)> t_DeltaUpdate;

	//! output from running reweighting scheme
	struct ReweightedResult {
		//! Number of iterations (outer loop)
		t_uint niters;
		//! Wether convergence was achieved
		bool good;
		//! Weights at last iteration
		WeightVector weights;
		//! Result from last inner loop
		typename Algorithm::DiagnosticAndResult algo;
		//! Default construction
		ReweightedResult() : niters(0), good(false), weights(WeightVector::Ones(1)), algo() {}
	};

	Reweighted(Algorithm const &algo, t_SetDelta const &setdelta, t_Reweightee const &reweightee)
	: algo_(algo), setdelta_(setdelta), reweightee_(reweightee),
	  itermax_(std::numeric_limits<t_uint>::max()), min_delta_(0e0), is_converged_(),
	  update_delta_([](Real delta) { return 8e-1 * delta; }), decomp_(mpi::Decomposition(false)) {}

	//! Underlying "inner-loop" algorithm
	Algorithm &algorithm() { return algo_; }
	//! Underlying "inner-loop" algorithm
	Algorithm const &algorithm() const { return algo_; }
	//! Sets the underlying "inner-loop" algorithm
	Reweighted<Algorithm> &algorithm(Algorithm const &algo) {
		algo_ = algo;
		return *this;
	}
	//! Sets the underlying "inner-loop" algorithm
	Reweighted<Algorithm> &algorithm(Algorithm &&algo) {
		algo_ = std::move(algo);
		return *this;
	}


	//! Function to reset the delta in the algorithm
	t_SetDelta const &set_delta() const { return setdelta_; }
	//! Function to reset the weights in the algorithm
	Reweighted<Algorithm> &set_delta(t_SetDelta const &setdelta) const {
		setdelta_ = setdelta;
		return *this;
	}
	//! Sets the delta on the underlying algorithm
	void set_delta(Algorithm &algo, Real const &delta) const {
		return set_delta()(algo, delta);
	}

	//! Function that needs to be reweighted
	//! \details E.g. \f$Ψ^Tx\f$. Note that l1-norm is not applied here.
	Reweighted<Algorithm> &reweightee(t_Reweightee const &rw) {
		reweightee_ = rw;
		return *this;
	}
	//! Function that needs to be reweighted
	t_Reweightee const &reweightee() const { return reweightee_; }
	//! Forwards to the reweightee function
	XVector reweightee(XVector const &x) const { return reweightee()(algorithm(), x); }

	//! Maximum number of reweighted iterations
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
	mpi::Decomposition decomp() const { return decomp_; }
	Reweighted &decomp(mpi::Decomposition decomp) {
		decomp_ = decomp;
		return *this;
	}
	bool is_converged(XVector const &x) const { return is_converged() ? is_converged()(x) : false; }

	//! \brief Performs reweighting
	//! \details This overload will compute an initial result without initial weights set to one.
	template <class INPUT>
	typename std::enable_if<not(std::is_same<INPUT, typename Algorithm::DiagnosticAndResult>::value
			or std::is_same<INPUT, ReweightedResult>::value),
			ReweightedResult>::type
			operator()(INPUT const &input) const;
	//! \brief Performs reweighting
	//! \details This overload will compute an initial result without initial weights set to one.
	ReweightedResult operator()() const;
	//! Reweighted algorithm, from prior call to inner-algorithm
	ReweightedResult operator()(typename Algorithm::DiagnosticAndResult const &warm) const;
	//! Reweighted algorithm, from prior call to reweighting algorithm
	ReweightedResult operator()(ReweightedResult const &warm) const;

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
	//! Function to set delta
	t_SetDelta setdelta_;
	//! \brief Function that is subject to reweighting
	//! \details E.g. \f$Ψ^Tx\f$. Note that l1-norm is not applied here.
	t_Reweightee reweightee_;
	//! Maximum number of reweighted iterations
	t_uint itermax_;
	//! \brief Lower limit for delta
	Real min_delta_;
	//! Checks convergence
	t_IsConverged is_converged_;
	//! Updates delta at each turn
	t_DeltaUpdate update_delta_;
	//! Decomposition object
	mpi::Decomposition decomp_;
};

template <class ALGORITHM>
template <class INPUT>
typename std::
enable_if<not(std::is_same<INPUT, typename ALGORITHM::DiagnosticAndResult>::value
		or std::is_same<INPUT, typename Reweighted<ALGORITHM>::ReweightedResult>::value),
		typename Reweighted<ALGORITHM>::ReweightedResult>::type
		Reweighted<ALGORITHM>::operator()(INPUT const &input) const {
	Algorithm algo = algorithm();
	return operator()(algo(input));
}

template <class ALGORITHM>
typename Reweighted<ALGORITHM>::ReweightedResult Reweighted<ALGORITHM>::operator()() const {
	Algorithm algo = algorithm();
	return operator()(algo());
}

template <class ALGORITHM>
typename Reweighted<ALGORITHM>::ReweightedResult Reweighted<ALGORITHM>::
operator()(typename Algorithm::DiagnosticAndResult const &warm) const {
	ReweightedResult result;
	result.algo = warm;
	result.weights = result.algo.l1_weights;
	return operator()(result);
}

template <class ALGORITHM>
typename Reweighted<ALGORITHM>::ReweightedResult Reweighted<ALGORITHM>::
operator()(ReweightedResult const &warm) const {
	// Copies inner algorithm, so that operator() can be constant
	Algorithm algo(algorithm());
	ReweightedResult result(warm);
        
    // We update delta first here because the first call to reweighted runs the algo before it
    // actually gets to here, so we need to take that into account.
	Real delta = std::max(min_delta(), update_delta(warm.algo.delta));

	if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
		PSI_LOW_LOG("-   Initial delta: {}", delta);
		PSI_LOW_LOG("-   Re-weighting iteration: {}", warm.algo.current_reweighting_iter);
	}

	for(result.niters = warm.algo.current_reweighting_iter; result.niters <= itermax(); ++result.niters) {
		//! Remove the adaptive_epsilon_start restriction after the first re-weighting iteration
		//! This allows restarting from saved runs without forcing the restriction to be undertaken multiple times.
		if(result.niters >= 1){
			algo.algorithm().adaptive_epsilon_start(0);
		}
		if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
			PSI_HIGH_LOG("Re-weighting iteration {}/{} ", result.niters, itermax());
			PSI_HIGH_LOG("  - delta: {}", delta);
		}

		if(!decomp().parallel_mpi() or decomp().my_number_of_root_wavelets() != 0){
			if(decomp().my_root_wavelet_comm().size() != 1){
				XVector local_x;
				local_x = decomp().my_root_wavelet_comm().broadcast(result.algo.x, decomp().my_root_wavelet_comm().root_id());
				result.weights = delta / (delta + reweightee(local_x).array().abs());
			}else{
				result.weights = delta / (delta + reweightee(result.algo.x).array().abs());
			}
			result.algo.l1_weights = result.weights;
			result.algo.delta = delta;
			result.algo.current_reweighting_iter = result.niters;
		}
		result.algo = algo(result.algo);
		bool good_result = false;
		if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){

			if(is_converged(result.algo.x)) {
				PSI_MEDIUM_LOG("Re-weighting scheme did converge in {} iterations", result.niters);
				good_result = true;
			}
		}
		good_result = decomp().global_comm().broadcast(good_result, decomp().global_comm().root_id());
		if(good_result){
			result.good = true;
			break;
		}
		if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
			delta = std::max(min_delta(), update_delta(delta));
		}
		if(decomp().parallel_mpi() and decomp().my_frequencies()[0].wavelet_comm.size() != 1 and decomp().my_frequencies()[0].number_of_wavelets != 0){
			delta = decomp().my_frequencies()[0].wavelet_comm.broadcast(delta, decomp().my_frequencies()[0].wavelet_comm.root_id());
		}
	}
	// result is always good if no convergence function is defined
	if(not is_converged()){
		result.good = true;
	}else if(not result.good){
		if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
			PSI_ERROR("Re-weighting scheme did *not* converge in {} iterations", itermax());
		}
	}
	return result;
}

//! Factory function to create an l0-approximation by reweighting an l1 norm
template <class ALGORITHM>
Reweighted<ALGORITHM>
reweighted(ALGORITHM const &algo,
		typename Reweighted<ALGORITHM>::t_Delta const &set_delta,
		typename Reweighted<ALGORITHM>::t_Reweightee const &reweightee) {
	return {algo, set_delta, reweightee};
}

template <class SCALAR> class PrimalDualTimeBlocking;
template <class ALGORITHM> class PositiveQuadrant;
template <class T>
Eigen::CwiseUnaryOp<const details::ProjectPositiveQuadrant<typename T::Scalar>, const T>
positive_quadrant(Eigen::DenseBase<T> const &input);

template <class SCALAR>
Reweighted<PositiveQuadrant<PrimalDualTimeBlocking<SCALAR>>>
reweighted(PrimalDualTimeBlocking<SCALAR> &algo) {

	typedef typename real_type<SCALAR>::type Real;
	auto const posq = positive_quadrant(algo);
	typedef typename std::remove_const<decltype(posq)>::type Algorithm;
	typedef Reweighted<Algorithm> RW;
	auto const reweightee
	= [](Algorithm const &posq, typename RW::XVector const &x) -> typename RW::XVector {
		assert(x.size() > 0);
		return posq.algorithm().Psi().adjoint() * x;
	};

	auto const set_delta = [](Algorithm &posq, Real const &delta) -> void {
		posq.algorithm().delta(delta);
	};


	return {posq, set_delta, reweightee};
}

} // namespace algorithm
} // namespace psi
#endif
