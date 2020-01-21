#ifndef PSI_PRIMAL_DUAL_WIDEBAND_BLOCKING_H
#define PSI_PRIMAL_DUAL_WIDEBAND_BLOCKING_H

#include <iostream>
#include <limits> // for std::numeric_limits<double>::epsilon()
#include <ctime>
#include "psi/config.h"
#include <functional>
#include <limits>
#include <vector>
#include "psi/exception.h"
#include "psi/linear_transform.h"
#include "psi/logging.h"
#include "psi/types.h"
#include "psi/l1_proximal.h"
#include "psi/proximal.h"
#include "psi/maths.h"
#include "psi/utilities.h"
#include "psi/mpi/decomposition.h"
#include "psi/io.h"
#include "psi/forward_backward.h"


namespace psi {
namespace algorithm {

//! \brief Primal Dual Wideband Blocking method
//! \details This is a basic implementation of the primal dual method using a forward backwards algorithm.
//! \f$\min_{, y, z} f() + l(y) + h(z)\f$ subject to \f$Î¦x = y\f$, \f$Î¨^Hx = z\f$
template <class SCALAR> class PrimalDualWidebandBlocking {
public:
	//! Scalar type
	typedef SCALAR value_type;
	//! Scalar type
	typedef value_type Scalar;
	//! Real type
	typedef typename real_type<Scalar>::type Real;
	//! Type of the underlying vectors
	typedef Vector<Scalar> t_Vector;
	//! Type of the Î¨ and Î¨^H operations, as well as Î¦ and Î¦^H
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
		Vector<Vector<Real>> epsilon; // necessary for epsilon update
		Vector<Real> weightsL21;
		Vector<Real> weightsNuclear;
		//! Kappa(s). delta, and current_reweighting_iter are passed back to ensure that values read in from file on a
		//! checkpoint restore are preserved. This is because in the checkpoint restore
		//! process kappa(s) are not passed to the PD algorithm when it is called and delta
		//! and current_reweighting_iter need to be passed out for reweighting.
		Real kappa1;
		Real kappa2;
		Real kappa3;
		Real delta;
		int current_reweighting_iter;
	};

	//! Setups PrimalDualWidebandBlocking
	template <class T>
	PrimalDualWidebandBlocking(std::vector<std::vector<Vector<T>>> const &target, const t_uint &image_size, const Vector<Vector<Real>> &l2ball_epsilon, std::vector<std::vector<std::shared_ptr<const t_LinearTransform>>>& Phi, std::vector<std::vector<Vector<Real>>> const &Ui)
	: itermax_(std::numeric_limits<t_uint>::max()), is_converged_(),
	  mu_(1.), tau_(1.), kappa1_(1.), kappa2_(1.), kappa3_(1.),
	  levels_(std::vector<t_uint>(1)), global_levels_(1),
	  l21_proximal_weights_(Vector<Real>::Ones(1)), nuclear_proximal_weights_(Vector<Real>::Ones(image_size)),
	  Phi_(Phi),
	  Psi_(std::vector<t_LinearTransform>(1,linear_transform_identity<Scalar>())),
	  Psi_Root_(linear_transform_identity<Scalar>()),
	  residual_convergence_(1e-4), decomp_(psi::mpi::Decomposition(false)),
	  relative_variation_(1e-4), relative_variation_x_(1e-4), positivity_constraint_(true), update_epsilon_(false), lambdas_(Vector<t_real>::Ones(3)),
	  P_(20), adaptive_epsilon_start_(200), target_(target), delta_(1), image_size_(image_size), l2ball_epsilon_(l2ball_epsilon), n_channels_(l2ball_epsilon.size()),
	  current_reweighting_iter_(0), preconditioning_(false), Ui_(Ui), itermax_fb_(20), relative_variation_fb_(1e-8) {}
	virtual ~PrimalDualWidebandBlocking() {}


	// Macro helps define properties that can be initialized as in
	// auto pd  = PrimalDualWidebandBlocking<float>().prop0(value).prop1(value);
#define PSI_MACRO(NAME, TYPE)                                                                      \
		TYPE const &NAME() const { return NAME##_; }                                                     \
		PrimalDualWidebandBlocking<SCALAR> &NAME(TYPE const &NAME) {                                             \
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
	//! Number of channels
	PSI_MACRO(n_channels, t_uint);
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
	//! Number of dictionaries assigned for each frequency to this process for the wavelet operator
	PSI_MACRO(levels, std::vector<t_uint>);
	//! Number of dictionaries in the overall wavelet operator for the root frequency
	PSI_MACRO(global_levels, t_uint);
	//! Weights for the l21-norm regularization
	PSI_MACRO(l21_proximal_weights, Vector<Real>);
	//! Weights for the nuclear-norm regularization
	PSI_MACRO(nuclear_proximal_weights, Vector<Real>);
	//! l2 ball bound
	PSI_MACRO(l2ball_epsilon, Vector<Vector<Real>>);
	//! Threshold for the update of l2ball_epsilon
	PSI_MACRO(lambdas, Vector<t_real>);
	//! Threshold for the relative variation of the solution (triggers the update of l2ball_epsilon)
	PSI_MACRO(relative_variation_x, Real);
	//! Number of iterations after which l2ball_epsilon can be updated
	PSI_MACRO(P, t_uint);
	//! Number of iterations after which l2ball_epsilon can be updated
	PSI_MACRO(adaptive_epsilon_start, t_uint);
	//! A function verifying convergence
	PSI_MACRO(is_converged, t_IsConverged);
	//! Measurement operator
	PSI_MACRO(Phi, std::vector<std::vector<std::shared_ptr<const t_LinearTransform>>>);
	//! Analysis operator
	PSI_MACRO(Psi, std::vector<t_LinearTransform>);
	//! Analysis operator just for the root wavelets. This is required to allow parallelisation of the root wavelets separately
	//! from the standard wavelet parallelisation
	PSI_MACRO(Psi_Root, t_LinearTransform);
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
	//! Current re-weighting iteration. Needed to enable this to be checkpointed.
	PSI_MACRO(current_reweighting_iter, int);
	//! Re-weighting delta
	PSI_MACRO(delta, Real);
	//! Whether preconditioning or not
	PSI_MACRO(preconditioning, bool);
	//! Preconditioning vector (equivalent to a diagonal preconditioner)
	PSI_MACRO(Ui, std::vector<std::vector<Vector<Real>>>);
	//! Maximum number of inner iterations (projection onto the ellipsoid)
	PSI_MACRO(itermax_fb, t_uint);
	//! Relative variation (stopping criterion)
	PSI_MACRO(relative_variation_fb, Real);

#undef PSI_MACRO

	psi::mpi::Decomposition const &decomp() const { return decomp_; }
	//! Sets the decomposition object
	PrimalDualWidebandBlocking<SCALAR> &decomp(psi::mpi::Decomposition const &decomp) {
		decomp_ = decomp;
		return *this;
	}
	//! Vector of target measurements
	std::vector<std::vector<t_Vector>> const &target() const { return target_; }
	//! Sets the vector of target measurements
	template <class T> PrimalDualWidebandBlocking<T> &target(std::vector<EigenCell<T>> const &target) {
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
	// AJ TODO Fix comments above
	Diagnostic operator()(t_Matrix &out, std::tuple<t_Matrix, t_Matrix, t_Matrix, std::vector<std::vector<t_Vector>>, t_Matrix, std::vector<std::vector<t_Vector>>, Vector<Vector<Real>>, Vector<Real>, Vector<Real>> const &guess) {
		return operator()(out, std::get<0>(guess), std::get<1>(guess), std::get<2>(guess), std::get<3>(guess), std::get<4>(guess), std::get<5>(guess), std::get<6>(guess), std::get<7>(guess), std::get<8>(guess));

	}
	//! \brief Calls Primal Dual
	//! \param[in] guess: initial guess
	DiagnosticAndResult operator()(std::tuple<t_Matrix, t_Matrix, t_Matrix, std::vector<std::vector<t_Vector>>, t_Matrix, std::vector<std::vector<t_Vector>>, Vector<Vector<Real>>, Vector<Real>, Vector<Real>> const &guess) {
		DiagnosticAndResult result;

		total_epsilons = Vector<Vector<Real>>(decomp().global_number_of_frequencies());
		for(int i=0; i<decomp().global_number_of_frequencies(); i++){
			total_epsilons[i] = Vector<Real>(decomp().frequencies()[i].number_of_time_blocks);
		}

		// Need to refactor the operator to allow this to happen without the explicit copy between guess and result below.
		result.p = guess.p;
		result.u = guess.u;
		result.v = guess.v;
		result.x_bar = guess.x_bar;
		result.epsilon = guess.epsilon;
		result.weightsL21 = guess.weightsL21;
		result.weightsNuclear = guess.weightsNuclear;
		static_cast<Diagnostic &>(result) = operator()(result.x, guess.x, result.p, result.u, result.v, result.x_bar, guess.residual, result.epsilon, result.weightsL21, result.weightsNuclear);
		return result;
	}
	//! \brief Calls Primal Dual
	DiagnosticAndResult operator()() {
		DiagnosticAndResult result;

		total_epsilons = Vector<Vector<Real>>(decomp().global_number_of_frequencies());
		for(int i=0; i<decomp().global_number_of_frequencies(); i++){
			total_epsilons[i] = Vector<Real>(decomp().frequencies()[i].number_of_time_blocks);
		}

		//! Checkpoint the simulation so it can be restarted from file if required. Only the master process (global_comm().is_root())
		//! does the checkpointing. Checkpointing is decided by a function in the decomp object that returns true if we should
		//! checkpoint now, and false otherwise.
		if(decomp().restore_checkpoint()){
			auto check = psi::io::IO<Scalar>();
			std::string filename = "restart.dat";
			psi::io::IOStatus restore_status = check.restore_wideband_with_distribute(decomp(), filename, result.x, l2ball_epsilon_, l21_proximal_weights_, nuclear_proximal_weights_, kappa1_, kappa2_, kappa3_, n_channels(), image_size(), delta_, current_reweighting_iter_);
			if(restore_status != psi::io::IOStatus::Success){
				decomp().global_comm().abort("Problem restoring from checkpoint. Quitting.");
			}
			if(decomp().global_comm().is_root()){
				PSI_HIGH_LOG("Restored kappa1 is {}",kappa1());
				PSI_HIGH_LOG("Restored kappa2 is {}",kappa2());
				PSI_HIGH_LOG("Restored kappa3 is {}",kappa3());
				PSI_HIGH_LOG("Restored re-weighting delta is {}",delta());
				PSI_HIGH_LOG("Restored re-weighting current iteration is {}",current_reweighting_iter());

			}
			//! Accessing the decomp object directly here so we can update the data structures. Accessing through the
			//! method (i.e. decomp().) forces constness on the object, whereas here we need to be able to change it.
			decomp_.restore_complete();
		}


		std::tuple<t_Matrix, t_Matrix, t_Matrix, std::vector<std::vector<t_Vector>>, t_Matrix, std::vector<std::vector<t_Vector>>, Vector<Vector<Real>>, Vector<Real>, Vector<Real>> guess = initial_guess();
		result.p = std::get<1>(guess);
		result.u = std::get<2>(guess);
		result.v = std::get<3>(guess);
		result.x_bar = std::get<4>(guess);
		result.epsilon = std::get<6>(guess);
		if(decomp().checkpoint_restored()){
			result.epsilon = l2ball_epsilon();
			result.weightsL21 = l21_proximal_weights();
			result.weightsNuclear = nuclear_proximal_weights();
		}else{
			result.x = std::get<0>(guess);
			result.epsilon = std::get<6>(guess);
			result.weightsL21 = std::get<7>(guess);
			result.weightsNuclear = std::get<8>(guess);
		}
		result.kappa1 = kappa1();
		result.kappa2 = kappa2();
		result.kappa3 = kappa3();
		result.delta = delta();
		result.current_reweighting_iter = current_reweighting_iter();


		static_cast<Diagnostic &>(result) = operator()(result.x, result.x, result.p, result.u, result.v, result.x_bar,  std::get<5>(guess), result.epsilon, result.weightsL21, result.weightsNuclear);
		// AJ I'd ideally do the below as it involves less memory operations, but I need to investigate how you do this in C++ more thoroughly first.
		//    t_Vector initial_x, initial_residual;
		//   (initial_x, result.s, result.v, result.x_bar, initial_residual) = initial_guess();
		//   static_cast<Diagnostic &>(result) = operator()(result.x, initial_x, result.s, result.v, result.x_bar, initial_residual);
		return result;
	}
	//! Makes it simple to chain different calls to Primal Dual
	DiagnosticAndResult operator()(DiagnosticAndResult const &warmstart) {
		DiagnosticAndResult result = warmstart;

		total_epsilons = Vector<Vector<Real>>(decomp().global_number_of_frequencies());
		for(int i=0; i<decomp().global_number_of_frequencies(); i++){
			total_epsilons[i] = Vector<Real>(decomp().frequencies()[i].number_of_time_blocks);
		}

		// Need to refactor the operator to allow this to happen without the explicit copy between warmstart and result below.
		result.p = warmstart.p;
		result.u = warmstart.u;
		result.v = warmstart.v;
		result.x_bar = warmstart.x_bar;
		result.epsilon = warmstart.epsilon;
		result.weightsL21 = warmstart.weightsL21;
		result.weightsNuclear = warmstart.weightsNuclear;
		result.kappa1 = warmstart.kappa1;
		result.kappa2 = warmstart.kappa2;
		result.kappa3 = warmstart.kappa3;
		result.delta = warmstart.delta;
		result.current_reweighting_iter = warmstart.current_reweighting_iter;

		kappa1_ = result.kappa1;
		kappa2_ = result.kappa2;
		kappa3_ = result.kappa3;

		delta_ = result.delta;
		current_reweighting_iter_ = result.current_reweighting_iter;

		static_cast<Diagnostic &>(result) = operator()(result.x, warmstart.x, result.p, result.u, result.v, result.x_bar, warmstart.residual, result.epsilon, result.weightsL21, result.weightsNuclear);
		return result;
	}

	//! \brief Computes initial guess for x and the residual
	//! \details with y the vector of measurements
	//! - x = Φ^T y
	//! - residuals = Φ x - y
	std::tuple<t_Matrix, t_Matrix, t_Matrix, std::vector<std::vector<t_Vector>>, t_Matrix, std::vector<std::vector<t_Vector>>, Vector<Vector<Real>>, Vector<Real>, Vector<Real>> initial_guess(bool checkpoint_loading = false) const {
		std::tuple<t_Matrix, t_Matrix, t_Matrix, std::vector<std::vector<t_Vector>>, t_Matrix, std::vector<std::vector<t_Vector>>, Vector<Vector<Real>>, Vector<Real>, Vector<Real>> guess; // x, p, u, v, residual, epsilon, l21weights, nuclearweights
		if(!checkpoint_loading and (!decomp().parallel_mpi() or decomp().global_comm().is_root())){
			std::get<0>(guess) = t_Matrix::Zero(image_size(), n_channels()); // x
		}else{
			std::get<0>(guess) = t_Matrix::Zero(1, 1); // x
		}
		if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
			std::get<1>(guess) = t_Matrix::Zero(image_size(), n_channels()); // p
		}else{
			std::get<1>(guess) = t_Matrix::Zero(1, 1); // p
		}
		if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
			std::get<2>(guess) = std::numeric_limits<Scalar>::epsilon()*t_Matrix::Ones(image_size()*levels()[0], n_channels()); // u
		}else{
			std::get<2>(guess) = std::numeric_limits<Scalar>::epsilon()*t_Matrix::Ones(1, 1); // u
		}
		std::get<3>(guess) = std::vector<std::vector<t_Vector>>(target().size());        // v
		if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
			std::get<4>(guess) = t_Matrix::Zero(image_size(), n_channels()); // x_bar
		}else{
			std::get<4>(guess) = t_Matrix::Zero(1, 1);
		}
		std::get<5>(guess) = target();     // residual
		if(!checkpoint_loading){
			std::get<6>(guess) = l2ball_epsilon();                               // epsilon
			std::get<7>(guess) = l21_proximal_weights();                         // l21_proximal_weights
			std::get<8>(guess) = nuclear_proximal_weights();                     // nuclear_proximal_weights
		}
		for (int l = 0; l < target().size(); ++l){
			std::get<3>(guess)[l] = std::vector<t_Vector>(target()[l].size());  // v
			std::get<5>(guess)[l].reserve(target()[l].size());  // residual
			for (int b = 0; b < target()[l].size(); ++b){
				std::get<3>(guess)[l][b] = t_Vector::Zero(target()[l][b].rows()); // v
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
	Vector<Vector<Real>> total_epsilons;
	Vector<Real> global_l21_weights;
	Vector<Real> root_l21_weights;
	psi::mpi::Decomposition decomp_;

	// Records data structures that need to be communicated but don't change throughout the simulation
	bool indices_calculated = false;
	std::vector<std::vector<Vector<t_int>>> indices;
	std::vector<std::vector<Vector<t_int>>> global_indices;
	std::vector<std::vector<Vector<t_int>>> frequency_indices;
	int freq_root_number;
	int max_block_number;
	int global_max_block_number;

	void iteration_step(t_Matrix &out, std::vector<std::vector<t_Vector>> &residual, t_Matrix &p, t_Matrix &u, std::vector<std::vector<t_Vector>> &v, t_Matrix &x_bar, Vector<Real> &w_l21, Vector<Real> &w_nuclear);

	//! Checks input makes sense
	void sanity_check(t_Matrix const &x_guess, t_Matrix const &p_guess, t_Matrix const &u_guess, std::vector<std::vector<t_Vector>> const &v_guess, t_Matrix const &x_bar_guess, std::vector<std::vector<t_Vector>> const &res_guess, Vector<Vector<Real>> const &l2ball_epsilon_guess) const {
		if(levels().size() > 1){
			for(int f=1; f<decomp().my_number_of_frequencies(); f++){
				if(levels()[f] != levels()[f-1]){
					PSI_THROW("different numbers of wavelets assigned for different frequencies, not currently supported");
				}
			}
		}
		if((!decomp().parallel_mpi() or decomp().global_comm().is_root()) and (Phi()[0][0]->adjoint() * target()[0][0]).size() != x_guess.rows())
			PSI_THROW("target, adjoint measurement operator and input vector have inconsistent sizes");
		if(Phi().size() != target().size())
			PSI_THROW("target and vector of measurement operators have inconsistent sizes");
		if((!decomp().parallel_mpi() or decomp().global_comm().is_root()) and x_bar_guess.size() != x_guess.size())
			PSI_THROW("x_bar and x have inconsistent sizes");
		if(target().size() != res_guess.size())
			PSI_THROW("target and residual vector have inconsistent sizes");
		if((!decomp().parallel_mpi() or decomp().global_comm().is_root()) and x_guess.cols() != n_channels())
			PSI_THROW("target, measurement operator and input vector have inconsistent sizes");
		if((!decomp().parallel_mpi() or decomp().global_comm().is_root()) and p_guess.size() != x_guess.size())
			PSI_THROW("input vector and dual variable p have inconsistent sizes");
		if((!decomp().parallel_mpi() or decomp().global_comm().is_root()) and u_guess.size() != x_guess.size()*levels()[0])
			PSI_THROW("input vector, measurement operator and dual variable u have inconsistent sizes");
		if(v_guess.size() != target().size())
			PSI_THROW("target and dual variable v have inconsistent sizes");
		if(not static_cast<bool>(is_converged()))
			PSI_WARN("No convergence function was provided: algorithm will run for {} steps", itermax());
		if(l2ball_epsilon_guess.size() != target().size())
			PSI_THROW("target and l2ball_epsilon have inconsistent sizes");
		if((!decomp().parallel_mpi() or decomp().global_comm().is_root()) and (l21_proximal_weights().size() != image_size()*levels()[0])){
			PSI_THROW("l21 proximal weights not the correct size");
		}
		if((!decomp().parallel_mpi() or decomp().global_comm().is_root()) and (nuclear_proximal_weights().size() != n_channels()))
			PSI_THROW("nuclear proximal weights not the correct size");
	}

	//! \brief Calls Primal Dual
	//! \param[out] out: Output vector
	//! \param[in] guess: initial guess
	//! \param[in] residuals: initial residuals
	// AJ TODO Update comments above
	Diagnostic operator()(t_Matrix &out, t_Matrix const &guess, t_Matrix &p_guess, t_Matrix &u_guess, std::vector<std::vector<t_Vector>> &v_guess, t_Matrix &x_bar_guess, std::vector<std::vector<t_Vector>> const &res_guess, Vector<Vector<Real>> &l2ball_epsilon_guess, Vector<Real> &l21_proximal_weights, Vector<Real> &nuclear_proximal_weights);

};

template <class SCALAR>
void PrimalDualWidebandBlocking<SCALAR>::iteration_step(t_Matrix &out, std::vector<std::vector<t_Vector>> &residual, t_Matrix &p, t_Matrix &u, std::vector<std::vector<t_Vector>> &v, t_Matrix &x_bar, Vector<Real> &w_l21, Vector<Real> &w_nuclear) {

	t_Matrix prev_sol;
	clock_t time1, time2, time3, time31, time32, time33, time34, time4, time5, time6, time7, time8, time9, time10, time11, time12, time13, time14, time15;

	time1 = clock();
	// Calculate the fourier indices for each Phi and then collect them on the frequency root
	// to allow only the specific x_hat and out_hat each process requires to be broadcast to them.
	// This is only done once, because the indices never change throughout a given simulation.
	if(decomp().parallel_mpi() and not indices_calculated){
		indices = std::vector<std::vector<Vector<t_int>>>(decomp().my_number_of_frequencies());
		for(int f=0; f<decomp().my_number_of_frequencies(); f++){
			indices[f] = std::vector<Vector<t_int>>(decomp().my_frequencies()[f].number_of_time_blocks);
		}
		for(int f=0; f<decomp().my_number_of_frequencies(); f++){
			for(int t=0; t<decomp().my_frequencies()[f].number_of_time_blocks; t++){
				indices[f][t] = Phi()[f][t]->get_fourier_indices(); // to be transmitted to the master node
			}
		}


		freq_root_number= 0;

		frequency_indices = std::vector<std::vector<Vector<t_int>>>(decomp().my_number_of_frequencies());
		for(int f=0; f<decomp().my_number_of_frequencies(); f++){
			if(decomp().my_frequencies()[f].freq_comm.is_root()){
				frequency_indices[freq_root_number] = std::vector<Vector<t_int>>(decomp().my_frequencies()[f].number_of_time_blocks);
				freq_root_number++;
			}
		}

		int freq_index = 0;
		for(int f=0; f<decomp().my_number_of_frequencies(); f++){
			if(decomp().my_frequencies()[f].freq_comm.is_root()){
				for(int t=0; t<decomp().my_frequencies()[f].number_of_time_blocks; t++){
					frequency_indices[freq_index][t] = Phi()[f][t]->get_fourier_indices(); // to be transmitted to the master node
				}
				freq_index++;
			}
		}

		if(decomp().global_comm().is_root()){
			global_indices = std::vector<std::vector<Vector<t_int>>>(decomp().global_number_of_frequencies());

			for(int f=0; f<decomp().global_number_of_frequencies(); f++){
				global_indices[f] = std::vector<Vector<t_int>>(decomp().frequencies()[f].number_of_time_blocks);
			}

		}
		//! After this the global indices are all present on the root
		decomp().template collect_indices<Vector<t_int>>(indices, global_indices, true);

		max_block_number = 0;
		for(int f=0; f<decomp().my_number_of_frequencies(); f++){
			if(decomp().my_frequencies()[f].number_of_time_blocks > max_block_number){
				max_block_number = decomp().my_frequencies()[f].number_of_time_blocks;
			}
		}

		global_max_block_number = 0;
		for(int f=0; f<decomp().global_number_of_frequencies(); f++){
			if(decomp().frequencies()[f].number_of_time_blocks > global_max_block_number){
				global_max_block_number = decomp().frequencies()[f].number_of_time_blocks;
			}
		}

		indices_calculated = true;

	}

	time1 = clock() - time1;

	time2 = clock();

	if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
		prev_sol = out;
	}

	time2 = clock() - time2;

	// prev_sol needs to be distributed, as does out here. Only needs distributed once, then should be able to
	// update every iteration.

	// The below assumes parallelisation over frequency at the moment
	// When we parallelise over time as well x_bar, p and u will be smaller than
	// v, but the x_bar and p and u (out etc..) can be replicated across all processes
	// with the same frequency. Then in the end the output is a sum reduction per frequency.

	// p_t = p_t-1 + x_bar - prox_nuclear_norm(p_t-1 + x_bar)
	// tmp can be constructed in parallel but nuclear norm is still serial at the moment so do this all on one process
	// Assuming that x_bar has been gathered from the previous timestep here already
	// This requires a distribution and gathering of x_bar before the first iteration and also a gathering of x_bar at the
	// end of the iteration.

	time3 = clock();

	if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
		time31 = clock();
		t_Matrix tmp = p + x_bar;
		time31 = clock() - time31;
		time32 = clock();
		t_Matrix p_prox(out.rows(), out.cols());
		time32 = clock() - time32;

		time33 = clock();
		proximal::nuclear_norm(p_prox, tmp, w_nuclear);
		time33 = clock() - time33;
		// Here p is local to 1 process
		time34 = clock();
		p = tmp - p_prox;
		time34 = clock() - time34;
	}

	time3 = clock() - time3;

	// distributed p here
	// std::cout << p << "\n"; // ok
	// for debugging purposes
	// auto norm_g0 = p.norm();

	// Update dual variable u (sparsity in the wavelet domain)
	//! u_t[l] = u_t-1[l] + Psi.adjoint()*x_bar[l] - prox_l21(u_t-1[l] + Psi.adjoint()*x_bar[l])
	// Can be done in parallel
	// Psi is the same for each band
	// x_bar needs distributed

	time4 = clock();

	// TODO: Strictly speaking temp1_u only needs to exist on wavelet processes for each process. To save memory, refactoring this
	// to only create it as the size of frequencies I am a wavelet process in would be sensible.
	t_Matrix temp1_u(image_size()*levels()[0], decomp().my_number_of_frequencies());
	t_Matrix x_bar_local(image_size(), decomp().my_number_of_frequencies());
	decomp().template distribute_frequency_data<t_Matrix, Scalar>(x_bar_local, x_bar, true);
	for(int f=0; f<decomp().my_number_of_frequencies(); f++){
		if(!decomp().parallel_mpi() or decomp().my_frequencies()[f].number_of_wavelets != 0){
			//! TODO MAKE x_bar_local correct here, i.e make sure frequency masters have the correct x_bar_local here to distribute.
			//! TODO x_bar_local is the frequency column of x_bar for this frequency
			t_Vector x_bar_local_freq(image_size());
			if(decomp().my_frequencies()[f].wavelet_comm.is_root()){
				x_bar_local_freq = x_bar_local.col(f);
			}
			if(decomp().my_frequencies()[f].wavelet_comm.size() != 1){
				x_bar_local_freq = decomp().my_frequencies()[f].wavelet_comm.broadcast(x_bar_local_freq, decomp().my_frequencies()[f].wavelet_comm.root_id());
			}
			temp1_u.col(f) = (Psi()[f].adjoint() * x_bar_local_freq).eval();
		}
		// Collect the temp1_u.col entry for this frequency onto the frequency master
		if(decomp().parallel_mpi() and decomp().my_frequencies()[f].number_of_wavelets != 0){
			Vector<t_complex> temp_data = temp1_u.col(f);
			decomp().my_frequencies()[f].wavelet_comm.distributed_sum(temp_data,decomp().my_frequencies()[f].freq_comm.root_id());
			if(decomp().my_frequencies()[f].freq_comm.is_root()){
				temp1_u.col(f) = temp_data;
			}
		}
	}

	time4 = clock() - time4;

	time5 = clock();

	// At the moment l21_norm can only be split along rows, and the above calculation of temp1_u splits along columns
	// So we need to reduce temp1_u here to allow the l21 norm to be calculated below
	t_Matrix u_local(image_size()*levels()[0], decomp().my_number_of_frequencies());
	t_Matrix global_temp1_u;
	if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
		t_Matrix u_prox(u.rows(), u.cols());
		t_Matrix global_temp1_u(u.rows(), u.cols());
		//Receive temp1_u here on root and put it in global_temp1_u
		decomp().template collect_frequency_root_data<Scalar>(temp1_u, global_temp1_u);
		//TODO Check this is correct and doesn't have to be done by cols
		global_temp1_u = global_temp1_u + u;
		proximal::l21_norm(u_prox, global_temp1_u, w_l21);
		// Here U is local to a single process
		u = global_temp1_u - u_prox;
		//distribute U if in parallel
		decomp().template distribute_frequency_data<t_Matrix, Scalar>(u_local, u, true);
	}else{
		//send temp1_u to global root
		decomp().template collect_frequency_root_data<Scalar>(temp1_u, global_temp1_u);
		//Receive U as frequency root
		decomp().template distribute_frequency_data<t_Matrix, Scalar>(u_local, u, true);
	}

	time5 = clock() - time5;

	time6 = clock();

	Matrix<t_complex> temp2_u(image_size(), decomp().my_number_of_frequencies());
	// temp2_u can be done in parallel but will require U to be scattered at this point.
	// TODO: Make sure U is in the correct configuration here
	for(int f=0; f<decomp().my_number_of_frequencies(); f++){
		if(!decomp().parallel_mpi() or decomp().my_frequencies()[f].number_of_wavelets != 0){
			t_Vector u_local_freq(image_size()*levels()[0]);
			if(decomp().my_frequencies()[f].wavelet_comm.is_root()){
				u_local_freq = u_local.col(f);
			}
			u_local_freq =  decomp().my_frequencies()[f].wavelet_comm.broadcast(u_local_freq, decomp().my_frequencies()[f].wavelet_comm.root_id());
			temp2_u.col(f) = static_cast<Vector<t_complex>>(Psi()[f](u_local_freq));
		}
		// Collect the temp2_u.col entry for this frequency onto the frequency master
		if(decomp().parallel_mpi() and decomp().my_frequencies()[f].number_of_wavelets != 0){
			Vector<t_complex> temp_data = temp2_u.col(f);
			decomp().my_frequencies()[f].wavelet_comm.distributed_sum(temp_data,decomp().my_frequencies()[f].freq_comm.root_id());
			if(decomp().my_frequencies()[f].freq_comm.is_root()){
				temp2_u.col(f) = temp_data;
			}
		}
	}

	time6 = clock() - time6;

	std::vector<std::vector<Eigen::SparseMatrix<t_complex>>> x_hat_sparse(decomp().my_number_of_frequencies());
	for(int f=0; f<decomp().my_number_of_frequencies(); f++){
		x_hat_sparse[f] = std::vector<Eigen::SparseMatrix<t_complex>>(decomp().my_frequencies()[f].number_of_time_blocks);
	}
	std::vector<Eigen::SparseMatrix<t_complex>> x_hat_global;

	time7 = clock();

	// TODO Make sure x_bar is distributed to frequency roots here
	std::vector<Matrix<t_complex>> x_hat(decomp().my_number_of_frequencies());
	int freq_root_count = 0;
	for (int f=0; f<decomp().my_number_of_frequencies(); ++f){ //frequency channels
		// If frequency root for this frequency
		if(decomp().my_frequencies()[f].freq_comm.is_root()){
			Vector<t_complex> const temp_data = x_bar_local.col(f).template cast<t_complex>();
			auto const image_bar = Image<t_complex>::Map(temp_data.data(), Phi()[0][0]->imsizey(), Phi()[0][0]->imsizex());
			x_hat[freq_root_count] = Phi()[0][0]->FFT(image_bar);
			freq_root_count++;
		}
	}

	time7 = clock() - time7;

	int my_freq_index = 0;
	int root_freq_index = 0;

	int shapes[decomp().global_number_of_frequencies()][global_max_block_number][3];

	time8 = clock();

	// Send x_hat from frequency root to frequency time block owners.
	for (int f=0; f<decomp().my_number_of_frequencies(); ++f){ //frequency channels
		if(decomp().my_frequencies()[f].freq_comm.is_root()){
			decomp_.initialise_requests(f,decomp().my_frequencies()[f].number_of_time_blocks*4);
		}else{
			decomp().template receive_fourier_data<t_complex>(x_hat_sparse[my_freq_index], f, f, true);
			my_freq_index++;
		}

		if(decomp().my_frequencies()[f].freq_comm.is_root()){
			x_hat_global = std::vector<Eigen::SparseMatrix<t_complex>>(decomp().my_frequencies()[f].number_of_time_blocks);
			int array_size = Phi()[0][0]->oversample_factor() * Phi()[0][0]->imsizey() * Phi()[0][0]->oversample_factor() * Phi()[0][0]->imsizex();
#ifdef PSI_OPENMP
#pragma omp parallel for default(shared)
#endif
			for(int t=0;t<decomp().my_frequencies()[f].number_of_time_blocks;t++){
				x_hat_global[t] = Eigen::SparseMatrix<t_complex>(array_size, 1);
				x_hat_global[t].reserve(frequency_indices[root_freq_index][t].rows());
				for (int k=0; k<frequency_indices[root_freq_index][t].rows(); k++){
					x_hat_global[t].insert(frequency_indices[root_freq_index][t](k),0) = x_hat[f](frequency_indices[root_freq_index][t](k),0);
				}
				x_hat_global[t].makeCompressed();
			}
			root_freq_index++;
			int my_index = 0;
			bool used_this_freq = false;
			for(int t=0;t<decomp().my_frequencies()[f].number_of_time_blocks;t++){
				decomp_.template send_fourier_data<t_complex>(x_hat_sparse[my_freq_index], x_hat_global, &shapes[f][t][0], t, my_index, f, used_this_freq, false);
			}
			if(used_this_freq){
				my_freq_index++;
			}
		}

		if(decomp().my_frequencies()[f].freq_comm.is_root()){
			decomp_.wait_on_requests(f, decomp().my_frequencies()[f].number_of_time_blocks*4);
			decomp_.cleanup_requests(f);
		}
	}

	time8 = clock() - time8;

	std::vector<std::vector<t_Vector>> Git_v(decomp().my_number_of_frequencies());

	time9 = clock();

#ifdef PSI_OPENMP
#pragma omp parallel for default(shared)
#endif
	for(int f=0; f<decomp().my_number_of_frequencies(); f++){

		Git_v[f] = std::vector<t_Vector>(target()[f].size());
		for (int t=0; t<decomp().my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
			t_Vector temp;
			t_Vector v_prox;

			if(preconditioning()){
				temp = v[f][t] + (Phi()[f][t]->G_function(x_hat_sparse[f][t])).eval().cwiseProduct(static_cast<t_Vector>(Ui()[f][t]));
				algorithm::ForwardBackward<SCALAR> ellipsoid_prox = algorithm::ForwardBackward<SCALAR>(temp.cwiseQuotient(Ui()[f][t]).eval(), target()[f][t])
	                          																							.Ui(Ui()[f][t])
																														.itermax(itermax_fb())
																														.l2ball_epsilon(l2ball_epsilon_[f](t))
																														.relative_variation(relative_variation_fb());
				v_prox = ellipsoid_prox().x;
				v[f][t] = temp - v_prox.cwiseProduct(Ui()[f][t]);
			}
			else{
				temp = v[f][t] + (Phi()[f][t]->G_function(x_hat_sparse[f][t])).eval();
				auto l2ball_proximal = proximal::L2Ball<SCALAR>(l2ball_epsilon_[f](t));
				v_prox = (l2ball_proximal(0, temp - target()[f][t]) + target()[f][t]);
				v[f][t] = temp - v_prox;
			}
			Git_v[f][t] = Phi()[f][t]->G_function_adjoint(v[f][t]);
		}
	}

	time9 = clock() - time9;

	time10 = clock();


	// Update dual variable v (data fitting term)
	//! v_t = v_t-1 + Phi*x_bar - l2ball_prox(v_t-1 + Phi*x_bar)
	// Phi is specific to each band so can be done in parallel
	// x_bar should already be distributed.
	// target would have to be distributed across processes
	// v should be distributed across processes
	Matrix<t_complex> temp2_v = Matrix<t_complex>::Zero(image_size(), decomp().my_number_of_frequencies());
	for(int f=0; f<decomp().my_number_of_frequencies(); f++){
		for (int t = 0; t < decomp().my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
			Vector<t_complex> temp_data = Git_v[f][t].template cast<t_complex>();
			Matrix<t_complex>  v1 = Matrix<t_complex>::Map(temp_data.data(),
					Phi()[0][0]->oversample_factor()*Phi()[0][0]->imsizey(),
					Phi()[0][0]->oversample_factor()*Phi()[0][0]->imsizex());
			Image<t_complex> im_tmp = Phi()[0][0]->inverse_FFT(v1);
			Vector<t_complex> v_tmp = Vector<t_complex>::Map(im_tmp.data(), im_tmp.size(), 1);
			temp2_v.col(f) = temp2_v.col(f) + v_tmp;
		}
		// Collect the temp2_v.col entry for this frequency on to the frequency master
		Vector<t_complex> temp_data = temp2_v.col(f);
		decomp().my_frequencies()[f].freq_comm.distributed_sum(temp_data,decomp().my_frequencies()[f].freq_comm.root_id());
		if(decomp().my_frequencies()[f].freq_comm.is_root()){
			temp2_v.col(f) = temp_data;
		}
	}

	time10 = clock() - time10;

	// Update primal variable x
	// Can be parallelised but needs P to be distributed
	// When parallelised in time we need to do the local out.col(l) per time instance and then reduce to a single column per frequency, then reduce to the master process.
	// Or we can reduce the temp2_v to the frequency masters and do the update there. Need to decide which makes sense mathematically and from a data distribution point of view.
	//	for (int l = 0; l < out.cols(); ++l){
	//		out.col(l) = prev_sol.col(l) - tau()*(kappa1()*p.col(l) + temp2_u.col(l)*kappa2() + temp2_v.col(l)*kappa3());
	//		out.col(l) = psi::positive_quadrant(out.col(l));
	//		x_bar.col(l) = 2*out.col(l) - prev_sol.col(l);
	//	}
	//  temp2_u and temp2_v need to be collected to root by this point

	Matrix<t_complex> global_temp2_v;
	Matrix<t_complex> global_temp2_u;

	time11 = clock();

	if(decomp().global_comm().is_root()){
		global_temp2_v = Matrix<t_complex>(out.rows(), out.cols());
		global_temp2_u = Matrix<t_complex>(out.rows(), out.cols());
	}

	decomp().template collect_frequency_root_data<t_complex>(temp2_u, global_temp2_u);
	decomp().template collect_frequency_root_data<t_complex>(temp2_v, global_temp2_v);


	time11 = clock() - time11;

	time12 = clock();

	if(decomp().global_comm().is_root()){
		out = prev_sol - tau()*(kappa1()*p + global_temp2_u*kappa2() + global_temp2_v*kappa3());
		if(positivity_constraint()){
			out = psi::positive_quadrant(out);
		}
		x_bar = 2*out - prev_sol;
	}

	time12 = clock() - time12;

	time13 = clock();

	std::vector<std::vector<Eigen::SparseMatrix<t_complex>>> out_hat_sparse(decomp().my_number_of_frequencies());
	for(int f=0; f<decomp().my_number_of_frequencies(); f++){
		out_hat_sparse[f] = std::vector<Eigen::SparseMatrix<t_complex>>(decomp().my_frequencies()[f].number_of_time_blocks);
	}
	//TODO Optimisation to send everything at once rather than one frequency at a time
	//std::vector<std::vector<Eigen::SparseMatrix<t_complex>>> out_hat_global;
	std::vector<Eigen::SparseMatrix<t_complex>> out_hat_global;
	std::vector<Matrix<t_complex>> out_hat;

	if(decomp().global_comm().is_root()){
		out_hat = std::vector<Matrix<t_complex>>(decomp().global_number_of_frequencies());
#ifdef PSI_OPENMP
#pragma omp parallel for default(shared)
#endif
		for (int f = 0; f < decomp().global_number_of_frequencies(); ++f){ //frequency channels
			Vector<t_complex> const temp_data = out.col(f).template cast<t_complex>();
			auto const image_out = Image<t_complex>::Map(temp_data.data(), Phi()[0][0]->imsizey(), Phi()[0][0]->imsizex());
			out_hat[f] = Phi()[0][0]->FFT(image_out);
		}
	}

	time13 = clock() - time13;

	// TODO Optimisation to send everything at once rather than one frequency at a time
	//if(decomp().global_comm().is_root()){
	//	out_hat_global = std::vector<std::vector<Eigen::SparseMatrix<t_complex>>>(decomp().global_number_of_frequencies());
	//}

	time14 = clock();

	my_freq_index = 0;

	for(int f=0; f<decomp().global_number_of_frequencies(); f++){

		if(decomp().frequencies()[f].in_this_frequency or decomp().global_comm().is_root()){

			if(decomp().global_comm().is_root()){
				decomp_.initialise_requests(f,decomp().frequencies()[f].number_of_time_blocks*4);
				out_hat_global = std::vector<Eigen::SparseMatrix<t_complex>>(decomp().frequencies()[f].number_of_time_blocks);
				int array_size = Phi()[0][0]->oversample_factor() * Phi()[0][0]->imsizey() * Phi()[0][0]->oversample_factor() * Phi()[0][0]->imsizex();
#ifdef PSI_OPENMP
#pragma omp parallel for default(shared)
#endif
				for(int t=0;t<decomp().frequencies()[f].number_of_time_blocks;t++){
					out_hat_global[t] = Eigen::SparseMatrix<t_complex>(array_size, 1);
					out_hat_global[t].reserve(global_indices[f][t].rows());
					for (int k=0; k<global_indices[f][t].rows(); k++){
						out_hat_global[t].insert(global_indices[f][t](k),0) = out_hat[f](global_indices[f][t](k),0);
					}
					out_hat_global[t].makeCompressed();
				}
				int my_index = 0;
				bool used_this_freq = false;
				for(int t=0;t<decomp().frequencies()[f].number_of_time_blocks;t++){
					decomp_.template send_fourier_data<t_complex>(out_hat_sparse[my_freq_index], out_hat_global, &shapes[my_freq_index][t][0], t, my_index, f, used_this_freq, true);
				}
				if(used_this_freq){
					my_freq_index++;
				}
			}else{
				if(decomp().frequencies()[f].in_this_frequency){
					decomp().template receive_fourier_data<t_complex>(out_hat_sparse[my_freq_index], f, my_freq_index, true);
					my_freq_index++;
				}
			}
			if(decomp().global_comm().is_root()){
				decomp_.wait_on_requests(f, decomp().frequencies()[f].number_of_time_blocks*4);
				decomp_.cleanup_requests(f);
			}

		}
	}

	time14 = clock() - time14;

	time15 = clock();

	// Update the residue
	// Can be parallelised
	// For the time parallelisation this only needs to be done by the frequency masters
#ifdef PSI_OPENMP
#pragma omp parallel for default(shared)
#endif
	for (int f = 0; f < decomp().my_number_of_frequencies(); ++f){   // assume the data are order per blocks per channel
		for (int t = 0; t < decomp().my_frequencies()[f].number_of_time_blocks; ++t){
			residual[f][t] = (Phi()[f][t]->G_function(out_hat_sparse[f][t])).eval() - target()[f][t];
		}
	}

	time15 = clock() - time15;

	PSI_LOW_LOG("{} InTime: 1: {} 2: {} 3: {} ({} {} {} {}) 4: {} 5: {} 6: {} 7: {} 8: {} 9: {} 10: {} 11: {} 12: {} 13: {} 14: {} 15: {}",
			decomp().global_comm().rank(),
			(float)time1/CLOCKS_PER_SEC,
			(float)time2/CLOCKS_PER_SEC,
			(float)time3/CLOCKS_PER_SEC,
			(float)time31/CLOCKS_PER_SEC,
			(float)time32/CLOCKS_PER_SEC,
			(float)time33/CLOCKS_PER_SEC,
			(float)time34/CLOCKS_PER_SEC,
			(float)time4/CLOCKS_PER_SEC,
			(float)time5/CLOCKS_PER_SEC,
			(float)time6/CLOCKS_PER_SEC,
			(float)time7/CLOCKS_PER_SEC,
			(float)time8/CLOCKS_PER_SEC,
			(float)time9/CLOCKS_PER_SEC,
			(float)time10/CLOCKS_PER_SEC,
			(float)time11/CLOCKS_PER_SEC,
			(float)time12/CLOCKS_PER_SEC,
			(float)time13/CLOCKS_PER_SEC,
			(float)time14/CLOCKS_PER_SEC,
			(float)time15/CLOCKS_PER_SEC);

}


template <class SCALAR>
typename PrimalDualWidebandBlocking<SCALAR>::Diagnostic PrimalDualWidebandBlocking<SCALAR>::
operator()(t_Matrix &out, t_Matrix const &x_guess, t_Matrix &p_guess, t_Matrix &u_guess, std::vector<std::vector<t_Vector>> &v_guess, t_Matrix &x_bar_guess, std::vector<std::vector<t_Vector>> const &res_guess, Vector<Vector<Real>> &l2ball_epsilon_guess, Vector<Real> &l21_proximal_weights_guess, Vector<Real> &nuclear_proximal_weights_guess) {

	clock_t temptime, time1, time2, time3, time4, time5, time6, time7, time8, time9, time10, time11;

	time1 = 0;
	time2 = 0;
	time3 = 0;
	time4 = 0;
	time5 = 0;
	time6 = 0;
	time7 = 0;
	time8 = 0;
	time9 = 0;
	time10 = 0;
	time11 = 0;


	if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
		PSI_HIGH_LOG("Performing wideband primal-dual");
	}

	sanity_check(x_guess, p_guess, u_guess, v_guess, x_bar_guess, res_guess, l2ball_epsilon_guess);

	std::vector<std::vector<t_Vector>> residual = res_guess;

	// Check if there is a user provided convergence function
	bool const has_user_convergence = static_cast<bool>(is_converged());
	bool converged = false;

	out = x_guess;

	decomp().global_comm().barrier();

	time1 = clock();

	l2ball_epsilon_ = l2ball_epsilon_guess;
	l21_proximal_weights_ = l21_proximal_weights_guess;
	nuclear_proximal_weights_ = nuclear_proximal_weights_guess;

	t_uint niters(0);

	decomp().template collect_epsilons_wideband_blocking<Vector<Vector<Real>>>(l2ball_epsilon_, total_epsilons);

	if(!decomp().restore_checkpoint()){
		if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
			global_l21_weights = Vector<Real>(image_size()*decomp().global_number_of_root_wavelets());
		}

		if(decomp().parallel_mpi() and decomp().my_number_of_root_wavelets() != 0){
			root_l21_weights = Vector<Real>(image_size()*decomp().my_number_of_root_wavelets());
			decomp().template collect_l21_weights<Vector<Real>>(l21_proximal_weights_, global_l21_weights, image_size());
			decomp().template distribute_l21_weights<Vector<Real>>(root_l21_weights, global_l21_weights, image_size());
		}else if(!decomp().parallel_mpi()){
			global_l21_weights = l21_proximal_weights_;
		}
	}else{
		if(decomp().parallel_mpi() and decomp().my_number_of_root_wavelets() != 0){
			root_l21_weights = Vector<Real>(image_size()*decomp().my_number_of_root_wavelets());
			decomp().template distribute_l21_weights<Vector<Real>>(root_l21_weights, global_l21_weights, image_size());
		}else if(!decomp().parallel_mpi()){
			 l21_proximal_weights_ = global_l21_weights;
		}
	}
	// Lambda function to compute the wavelet-related regularization term
	auto wavelet_regularization = [](t_LinearTransform psi, const t_Matrix &X, const t_uint rows) {
		t_Matrix Y(rows, X.cols());
#ifdef PSI_OPENMP
#pragma omp parallel for default(shared)
#endif
		for (int l = 0; l < X.cols(); ++l){
			Y.col(l) = static_cast<t_Vector>(psi.adjoint() * X.col(l));
		}
		return Y;
	};

	decomp().global_comm().barrier();

	time1 = clock() - time1;

	std::pair<Real, Real> objectives;

	time2 = clock();

	t_Matrix partial;
	Real partial_sum;
	if(!decomp().parallel_mpi() or decomp().my_number_of_root_wavelets() != 0){
		t_Matrix local_out;
		if(decomp().my_root_wavelet_comm().size() != 1){
			local_out = decomp().my_root_wavelet_comm().broadcast(out, decomp().global_comm().root_id());
			partial = wavelet_regularization(Psi_Root(), local_out, image_size()*decomp().my_number_of_root_wavelets());
		}else{
			partial = wavelet_regularization(Psi_Root(), out, image_size()*decomp().my_number_of_root_wavelets());
		}
	    partial_sum = psi::l21_norm(partial, root_l21_weights) * mu();

		if(decomp().parallel_mpi() and decomp().my_root_wavelet_comm().size() != 1){
			decomp().my_root_wavelet_comm().distributed_sum(&partial_sum, decomp().global_comm().root_id());
		}
	}

	decomp().global_comm().barrier();

	time2 = clock() - time2;

	time3 = clock();

	if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
		// This can be parallelised, it would have to be done across the frequency 0 wavelet workers, but then they'd need to know their respective columns
		// of out, so it'd need some communication. So for now, do it all on the root. This requires a full, not parallelised, Psi().
		objectives.first = psi::nuclear_norm(out, nuclear_proximal_weights()) + partial_sum;
		objectives.second = 0.;
	}

	decomp().global_comm().barrier();

	time3 = clock() - time3;

	time4 = clock();

	Vector<Vector<Real>> residual_norm = Vector<Vector<Real>>(decomp().my_number_of_frequencies());
	for(int f=0; f<decomp().my_number_of_frequencies(); f++){
		residual_norm[f] = Vector<Real>::Zero(decomp().my_frequencies()[f].number_of_time_blocks);
	}

	Vector<Vector<int>> counter = Vector<Vector<int>>::Zero(decomp().my_number_of_frequencies());
	for(int f=0; f<decomp().my_number_of_frequencies(); f++){
		counter[f] = Vector<int>::Zero(decomp().my_frequencies()[f].number_of_time_blocks);
	}

	Vector<Real> w_l21;
	Vector<Real> w_nuclear;

	if(!decomp().parallel_mpi() or decomp().my_frequencies()[0].number_of_wavelets != 0){
		w_l21 = mu()*l21_proximal_weights()/kappa2();
	}

	if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
		w_nuclear = nuclear_proximal_weights()/kappa1();
	}

	decomp().global_comm().barrier();

	time4 = clock() - time4;

	for(; niters < itermax(); ++niters) {

		t_Matrix x_old;
		// Currently x_old is only on the root process as with out
		if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
			x_old = out;
			PSI_HIGH_LOG("    - Iteration {}/{}", niters, itermax());
		}

		temptime = clock();

		iteration_step(out, residual, p_guess, u_guess, v_guess, x_bar_guess, w_l21, w_nuclear);

		decomp().global_comm().barrier();

		temptime = clock() - temptime;
		time5 += temptime;

		temptime = clock();

		for(int f=0; f<decomp().my_number_of_frequencies(); ++f){
			for(int t=0;t<decomp().my_frequencies()[f].number_of_time_blocks;t++){
				residual_norm[f][t] = residual[f][t].stableNorm();
			}
		}

		Vector<Vector<Real>> total_residual_norms = Vector<Vector<Real>>(decomp().global_number_of_frequencies());
		for(int f=0; f<decomp().global_number_of_frequencies(); f++){
			total_residual_norms[f] = Vector<Real>(decomp().frequencies()[f].number_of_time_blocks);
		}
		decomp().template collect_residual_norms<Real>(residual_norm, total_residual_norms);

		decomp().global_comm().barrier();

		temptime = clock() - temptime;
		time6 += temptime;

		temptime = clock();

		if(!decomp().parallel_mpi() or decomp().my_number_of_root_wavelets() != 0){
			t_Matrix local_out;
			if(decomp().my_root_wavelet_comm().size() != 1){
				local_out = decomp().my_root_wavelet_comm().broadcast(out, decomp().global_comm().root_id());
				partial = wavelet_regularization(Psi_Root(), local_out, image_size()*decomp().my_number_of_root_wavelets());
			}else{
				partial = wavelet_regularization(Psi_Root(), out, image_size()*decomp().my_number_of_root_wavelets());
			}
			partial_sum = psi::l21_norm(partial, root_l21_weights) * mu();

			if(decomp().parallel_mpi() and decomp().my_root_wavelet_comm().size() != 1){
				decomp().my_root_wavelet_comm().distributed_sum(&partial_sum, decomp().global_comm().root_id());
			}
		}

		decomp().global_comm().barrier();

		temptime = clock() - temptime;
		time7 += temptime;

		if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){

			for(int f=0; f<decomp().global_number_of_frequencies(); f++){
				PSI_MEDIUM_LOG("      - Sum of residuals[{}]: {}", f, total_residual_norms[f].sum());
			}

			temptime = clock();

			// update objective function
			objectives.second = objectives.first;

			temptime = clock() - temptime;
			time8 += temptime;

			temptime = clock();

			objectives.first = psi::nuclear_norm(out, nuclear_proximal_weights()) + partial_sum;
			t_real const relative_objective = std::abs(objectives.first - objectives.second) / objectives.first;
			PSI_HIGH_LOG("    - objective: obj value = {}, rel obj = {}", objectives.first,
					relative_objective);

			temptime = clock() - temptime;
			time9 += temptime;

			temptime = clock();

			Real total_residual_norm = 0;
			Real l2ball_epsilon_norm = 0;
			for(int f=0; f<decomp().global_number_of_frequencies(); f++){
				l2ball_epsilon_norm += total_epsilons[f].squaredNorm();
				PSI_LOW_LOG("Debugging residual norm {} {} {}",f,total_residual_norms[f].squaredNorm(),total_residual_norm);
				total_residual_norm +=  total_residual_norms[f].squaredNorm();
				PSI_LOW_LOG("Debugging 2 residual norm {} {} {}",f,total_residual_norms[f].squaredNorm(),total_residual_norm);
			}
			total_residual_norm = std::sqrt(total_residual_norm);
			l2ball_epsilon_norm = std::sqrt(l2ball_epsilon_norm);

			temptime = clock() - temptime;
			time10 += temptime;

			PSI_HIGH_LOG("      -  residual norm = {}, residual convergence = {}", total_residual_norm, residual_convergence() * (l2ball_epsilon_norm));

			auto const user = (not has_user_convergence) or is_converged(out);
			auto const res = (residual_convergence() <= 0e0) or (total_residual_norm < residual_convergence() * (l2ball_epsilon_norm));
			auto const rel = relative_variation() <= 0e0 or relative_objective < relative_variation();

			converged = user and res and rel;

		}

		decomp().global_comm().barrier();


		if(decomp().parallel_mpi()){
			int temp_converged;
			// Setting the value to be broadcast. This is done on all processes
			// but the value only matters on the root process
			if(converged){
				temp_converged = 1;
			}else{
				temp_converged = 0;
			}
			temp_converged = decomp().global_comm().broadcast(temp_converged, decomp().global_comm().root_id());
			if(temp_converged == 1){
				converged = true;
			}else{
				converged = false;
			}
		}

		//! Checkpoint the simulation so it can be restarted from file if required. Only the master process (global_comm().is_root())
		//! does the checkpointing. Checkpointing is decided by a function in the decomp object that returns true if we should
		//! checkpoint now, and false otherwise.
		if(converged or decomp().checkpoint_now(niters, itermax())){
			if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
				PSI_HIGH_LOG("Checkpointing {}/{}",niters,itermax());
			}
			std::string filename = "restart.dat";
			auto check = psi::io::IO<Scalar>();
			psi::io::IOStatus checkpoint_status = check.checkpoint_wideband_with_collect(decomp(), filename, out, l2ball_epsilon(), global_l21_weights, nuclear_proximal_weights(),  kappa1(), kappa2(),  kappa3(), image_size(), delta(), current_reweighting_iter());
			if((!decomp().parallel_mpi() or decomp().global_comm().is_root()) and checkpoint_status != psi::io::IOStatus::Success){
				PSI_HIGH_LOG("Problem checkpointing to file, error is: {}", psi::io::IO<Scalar>::GetErrorMessage(checkpoint_status));
			}
		}

		// Only print on the root
		if(converged) {
			if(!decomp().parallel_mpi() || decomp().global_comm().is_root()){
				PSI_HIGH_LOG("    - converged in {} of {} iterations", niters, itermax());
			}
			break;
		}

		bool rel_x_check;

		if(!decomp().parallel_mpi() || decomp().global_comm().is_root()){
			// update epsilon
			// Only done on the root process
			auto const rel_x = (out - x_old).norm();

			rel_x_check = (rel_x < relative_variation_x()*out.norm());
		}

		// Need to broadcast rel_x_check here
		rel_x_check = decomp().global_comm().broadcast(rel_x_check, decomp().global_comm().root_id());

		temptime = clock();

		// update l2ball_epsilon (if appropriate, only for real data -> add a boolean variable to control this feature)
		// As we already have out gathered at this point, we can just do this in serial. If the out nuclear norm is
		// parallelised in the future we can parallelise this.
		if(update_epsilon() && niters > adaptive_epsilon_start() && rel_x_check){
			bool eps_updated = false;
			for(int f=0; f<residual_norm.size(); ++f){
				for(int t=0; t<residual_norm[f].size(); ++t){
					bool eps_flag = residual_norm[f][t] < lambdas()(0) * l2ball_epsilon_[f][t] or residual_norm[f][t] > lambdas()(1) * l2ball_epsilon_[f][t];
					if(niters > counter[f](t) + P() and eps_flag){
						PSI_HIGH_LOG("Update epsilon for block {}, channel {}, at iteration {}",t,f,niters);
						PSI_LOW_LOG("epsilon before: {}, residual norm: {} ",l2ball_epsilon_[f][t],residual_norm[f](t));
						l2ball_epsilon_[f](t) = lambdas()(2) * residual_norm[f][t] + (1 - lambdas()(2)) * l2ball_epsilon_[f][t];
						PSI_LOW_LOG("epsilon after: {}",l2ball_epsilon_[f][t]);
						counter[f](t) = niters;
						eps_updated = true;
					}
				}
			}
			decomp().template collect_epsilons_wideband_blocking<Vector<Vector<Real>>>(l2ball_epsilon(), total_epsilons);
			if(eps_updated and (!decomp().parallel_mpi() or decomp().global_comm().is_root())){
				Real epsilon_norm = 0;
				Real epsilon_l2_norm = 0;

				for(int f=0; f<decomp().global_number_of_frequencies(); f++){
					epsilon_norm += total_epsilons[f].norm();
					epsilon_l2_norm += total_epsilons[f].stableNorm();

				}
				PSI_HIGH_LOG("sum of epsilon norms: {}",epsilon_norm);
				PSI_HIGH_LOG("sum of epsilon l2 norm: {}",epsilon_l2_norm);
			}
		}

		decomp().global_comm().barrier();

		temptime = clock() - temptime;
		time11 += temptime;
	}

	PSI_HIGH_LOG("{} OutTime: 1: {} 2: {} 3: {} 4: {} 5: {} 6: {} 7: {} 8: {} 9: {} 10: {} 11: {}",
			decomp().global_comm().rank(),
			(float)time1/CLOCKS_PER_SEC,
			(float)time2/CLOCKS_PER_SEC,
			(float)time3/CLOCKS_PER_SEC,
			(float)time4/CLOCKS_PER_SEC,
			(float)time5/CLOCKS_PER_SEC,
			(float)time6/CLOCKS_PER_SEC,
			(float)time7/CLOCKS_PER_SEC,
			(float)time8/CLOCKS_PER_SEC,
			(float)time9/CLOCKS_PER_SEC,
			(float)time10/CLOCKS_PER_SEC,
			(float)time11/CLOCKS_PER_SEC);



	// check function exists, otherwise, don't know if convergence is meaningful
	if(not converged){
		if(!decomp().parallel_mpi() || decomp().global_comm().is_root()){
			PSI_ERROR("    - did not converge within {} iterations", itermax());
		}
	}

	l2ball_epsilon_guess = l2ball_epsilon_;

	return {niters, converged, std::move(residual)};
} /* end of operator */
} /* psi::algorithm */
} /* psi */
#endif
