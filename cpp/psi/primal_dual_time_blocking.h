#ifndef PSI_PRIMAL_DUAL_TIME_BLOCKING_H
#define PSI_PRIMAL_DUAL_TIME_BLOCKING_H

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
#include "psi/forward_backward.h"
#include "psi/io.h"

namespace psi {
namespace algorithm {

//! \brief Primal Dual method
//! \details This is a basic implementation of the primal dual method using a forward backwards algorithm.
//! \f$\min_{, y, z} f() + l(y) + h(z)\f$ subject to \f$Φx = y\f$, \f$Ψ^Hx = z\f$
template <class SCALAR> class PrimalDualTimeBlocking {
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
	typedef psi::Matrix<Scalar> t_Matrix;
	//! Values indicating how the algorithm ran
	struct Diagnostic {
		//! Number of iterations
		t_uint niters;
		//! Whether convergence was achieved
		bool good;
		//! the residual from the last iteration
		std::vector<t_Vector> residual;

		Diagnostic(t_uint niters = 0u, bool good = false)
		: niters(niters), good(good), residual({t_Vector::Zero(1)}) {}
		Diagnostic(t_uint niters, bool good, std::vector<t_Vector> &&residual)
		: niters(niters), good(good), residual(std::move(residual)) {}
	};
	//! Holds result vector as well
	struct DiagnosticAndResult : public Diagnostic {
		//! Output
		t_Vector x;
		//! Dual variables u and v
		t_Vector u;
		t_Vector x_bar;
		std::vector<t_Vector> v;
		//! epsilon parameter (updated for real data)
		Vector<Real> epsilon; // necessary for epsilon update
		//! l1 proximal weights
		Vector<Real> l1_weights;
		//! Kappa, sigma2, delta, and current_reweighting_iter are passed back to ensure that values read in from file on a
		//! checkpoint restore are preserved. This is because in the checkpoint restore
		//! process kappa and sigma2 are not passed to the PD algorithm when it is called and delta
		//! and current_reweighting_iter need to be passed out for reweighting.
		Real kappa;
		Real sigma2;
		Real delta;
		int current_reweighting_iter;
	};

	//! Setups PrimalDualTimeBlocking
	// If in parallel target, Phi, and l2ball_epsilon need to be distributed appropriately

	template <class T>
	PrimalDualTimeBlocking(EigenCell<T> const &target, const t_uint &image_size, const Vector<Real> &l2ball_epsilon, std::vector<std::shared_ptr<const t_LinearTransform>>& Phi, std::vector<Vector<Real>> const &Ui)
	: itermax_(std::numeric_limits<t_uint>::max()), is_converged_(), kappa_(1.), tau_(0.49), sigma1_(1.),
	  sigma2_(1.), levels_(1), l2ball_epsilon_(l2ball_epsilon), l1_proximal_weights_(Vector<Real>::Zero(1)),
	  Phi_(Phi), Psi_(linear_transform_identity<Scalar>()), image_size_(image_size),
	  residual_convergence_(1.001), relative_variation_(1e-4), relative_variation_x_(1e-4), positivity_constraint_(true),
	  update_epsilon_(false), lambdas_(Vector<t_real>::Ones(3)), P_(20), adaptive_epsilon_start_(200),
	  target_(target), decomp_(psi::mpi::Decomposition(false)), preconditioning_(false), Ui_(Ui), itermax_fb_(20), relative_variation_fb_(1e-8),
	  delta_(1), current_reweighting_iter_(0) {}
	virtual ~PrimalDualTimeBlocking() {}

#define PSI_MACRO(NAME, TYPE)                                                                     \
		TYPE const &NAME() const { return NAME##_; }                                              \
		PrimalDualTimeBlocking<SCALAR> &NAME(TYPE const &NAME) {                                  \
			NAME##_ = NAME;                                                                       \
			return *this;                                                                         \
		}                                                                                         \
		\
protected:                                                                                        \
TYPE NAME##_;                                                                                     \
\
public:

	//! Maximum number of iterations
	PSI_MACRO(itermax, t_uint);
	//! Image size
	PSI_MACRO(image_size, t_uint);
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
	//! l2 ball bound
	PSI_MACRO(l2ball_epsilon, Vector <t_real>);
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
	PSI_MACRO(Phi, std::vector<std::shared_ptr<const t_LinearTransform>>);
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
	//! Whether preconditioning or not
	PSI_MACRO(preconditioning, bool);
	//! Preconditioning vector (equivalent to a diagonal preconditioner)
	PSI_MACRO(Ui, std::vector<Vector<Real>>);
	//! Maximum number of inner iterations (projection onto the ellipsoid)
	PSI_MACRO(itermax_fb, t_uint);
	//! Relative variation (stopping criterion)
	PSI_MACRO(relative_variation_fb, Real);
	//! Re-weighting delta
	PSI_MACRO(delta, Real);
	//! Current re-weighting iterations
	PSI_MACRO(current_reweighting_iter, int);


#undef PSI_MACRO

	psi::mpi::Decomposition const &decomp() const { return decomp_; }
	//! Sets the decomposition object
	PrimalDualTimeBlocking<SCALAR> &decomp(psi::mpi::Decomposition const &decomp) {
		decomp_ = decomp;
		return *this;
	}
	//! Vector of target measurements
	std::vector<t_Vector> const &target() const { return target_; }
	//! Sets the vector of target measurements
	template <class T> PrimalDualTimeBlocking<T> &target(EigenCell<T> const &target) {
		target_ = target;
		return *this;
	}



	//! Facilitates call to user-provided convergence function
	bool is_converged(t_Vector const &x) const {
		return static_cast<bool>(is_converged()) and is_converged()(x);
	}


	//! \brief Calls Primal Dual
	//! \param[out] out: Output vector x
	Diagnostic operator()(t_Vector &out) const { return operator()(out, initial_guess()); }
	//! \brief Calls Primal Dual
	//! \param[out] out: Output vector x
	//! \param[in] guess: initial guess
	// AJ TODO Fix comments above
	//! \brief Calls Primal Dual
	Diagnostic operator()(t_Vector &out, std::tuple<t_Vector, t_Vector, std::vector<t_Vector>, t_Vector, std::vector<t_Vector>, Vector<Real>, Vector<Real>> const &guess) {
		return operator()(out, std::get<0>(guess), std::get<1>(guess), std::get<2>(guess), std::get<3>(guess), std::get<4>(guess), std::get<5>(guess), std::get<6>(guess));
	}

	//! \brief Calls Primal Dual
	//! \param[in] guess: initial guess
	DiagnosticAndResult operator()(std::tuple<t_Vector, t_Vector, std::vector<t_Vector>, t_Vector, std::vector<t_Vector>, Vector<Real>, Vector<Real>> const &guess) {
		PSI_HIGH_LOG("Guess operator");
		DiagnosticAndResult result;
		// Need to refactor the operator to allow this to happen without the explicit copy between guess and result below.
		result.u = guess.u;
		result.v = guess.v;
		result.x_bar = guess.x_bar;
		result.epsilon = guess.epsilon;
		result.l1_weights = guess.l1_weights;
		static_cast<Diagnostic &>(result) = operator()(result.x, guess.x, result.u, result.v, result.x_bar, guess.residual, result.epsilon, result.l1_weights);
		return result;
	}


	//! \brief Calls Primal Dual
	//! \param[in] guess: initial guess
	DiagnosticAndResult operator()() {

		PSI_HIGH_LOG("Initial operator");

		DiagnosticAndResult result;
		total_epsilons = Vector<Real>(decomp().frequencies()[0].number_of_time_blocks);

		//! Checkpoint the simulation so it can be restarted from file if required. Only the master process (global_comm().is_root())
		//! does the checkpointing. Checkpointing is decided by a function in the decomp object that returns true if we should
		//! checkpoint now, and false otherwise.
		if(decomp().restore_checkpoint()){
			auto check = psi::io::IO<Scalar>();
			std::string filename = "restart.dat";
			psi::io::IOStatus restore_status = check.restore_time_blocking_with_distribute(decomp(), filename, result.x, l2ball_epsilon_, l1_proximal_weights_, kappa_, sigma2_, image_size(), delta_, current_reweighting_iter_);
			if(restore_status != psi::io::IOStatus::Success){
				decomp().global_comm().abort("Problem restoring from checkpoint. Quitting.");
			}
			if(decomp().global_comm().is_root()){
				PSI_HIGH_LOG("Restored kappa is {}",kappa());
				PSI_HIGH_LOG("Restored sigma2 is {}",sigma2());
				PSI_HIGH_LOG("Restored re-weighting delta is {}",delta());
				PSI_HIGH_LOG("Restored re-weighting current iteration is {}",current_reweighting_iter());

			}
			//! Accessing the decomp object directly here so we can update the data structures. Accessing through the
			//! method (i.e. decomp().) forces constness on the object, whereas here we need to be able to change it.
			decomp_.restore_complete();
		}

		std::tuple<t_Vector, t_Vector, std::vector<t_Vector>, t_Vector, std::vector<t_Vector>, Vector<Real>, Vector<Real>> guess = initial_guess(decomp().checkpoint_restored());

		result.u = std::get<1>(guess);
		result.v = std::get<2>(guess);
		result.x_bar = std::get<3>(guess);
		if(decomp().checkpoint_restored()){
			result.epsilon = l2ball_epsilon();
			result.l1_weights = l1_proximal_weights();
		}else{
			result.x = std::get<0>(guess);
			result.epsilon = std::get<5>(guess);
			result.l1_weights = std::get<6>(guess);
		}
		result.kappa = kappa();
		result.sigma2 = sigma2();
		result.delta = delta();
		result.current_reweighting_iter = current_reweighting_iter();

		static_cast<Diagnostic &>(result) = operator()(result.x, result.x, result.u, result.v, result.x_bar,  std::get<4>(guess), result.epsilon, result.l1_weights);
		// AJ I'd ideally do the below as it involves less memory operations, but I need to investigate how you do this in C++ more thoroughly first.
		//    t_Vector initial_x, initial_residual;
		//   (initial_x, result.s, result.v, result.x_bar, initial_residual) = initial_guess();
		//   static_cast<Diagnostic &>(result) = operator()(result.x, initial_x, result.s, result.v, result.x_bar, initial_residual);
		return result;
	}
	//! Makes it simple to chain different calls to Primal Dual
	DiagnosticAndResult operator()(DiagnosticAndResult const &warmstart) {
		total_epsilons = Vector<Real>(decomp().frequencies()[0].number_of_time_blocks);
		DiagnosticAndResult result = warmstart;
		// Need to refactor the operator to allow this to happen without the explicit copy between warmstart and result below.
		// Only on wavelet process(es)
		result.u = warmstart.u;
		// V should already be distributed. Need to make sure it is still distributed
		result.v = warmstart.v;

		// Only on root
		result.x_bar = warmstart.x_bar;
		// epsilon distributed per Phi.
		result.epsilon = warmstart.epsilon;

		result.l1_weights = warmstart.l1_weights;

		result.kappa = warmstart.kappa;
		result.sigma2 = warmstart.sigma2;
		result.delta = warmstart.delta;
		result.current_reweighting_iter = warmstart.current_reweighting_iter;

		kappa_ = result.kappa;
		sigma2_ = result.sigma2;
		delta_ = result.delta;
		current_reweighting_iter_ = result.current_reweighting_iter;

		// residual per Phi, distributed as epsilon
		static_cast<Diagnostic &>(result) = operator()(result.x, warmstart.x, result.u, result.v, result.x_bar, warmstart.residual, result.epsilon, result.l1_weights);
		return result;
	}


	//! \brief Computes initial guess for x and the residual
	//! \details with y the vector of measurements
	//! - x = Phi^T y
	//! - residuals = Phi x - y
	std::tuple<t_Vector, t_Vector, std::vector<t_Vector>, t_Vector, std::vector<t_Vector>, Vector<Real>, Vector<Real>> initial_guess(bool checkpoint_loading = false) const {
		std::tuple<t_Vector, t_Vector, std::vector<t_Vector>, t_Vector, std::vector<t_Vector>, Vector<Real>, Vector<Real>> guess; // x, u, v, x_bar, residual, epsilon, l1_proximal_weights
		if(!checkpoint_loading){
			// Only on root process
			if(decomp().global_comm().is_root()){
				std::get<0>(guess) = t_Vector::Zero(image_size()); 				// x
			}else{
				std::get<0>(guess) = t_Vector::Zero(1); 				// x
			}
		}
		// Distributed across wavelet process(es)
		std::get<1>(guess) = t_Vector::Zero(image_size()*levels()); 	// u
		// Distributed v
		std::get<2>(guess) = std::vector<t_Vector>(target().size());    // v
		// Only on root process
		std::get<3>(guess) = t_Vector::Zero(image_size()); 				// x_bar
		// Distributed with Phis
		std::get<4>(guess) = std::vector<t_Vector>(Phi().size());       // residual
		if(!checkpoint_loading){
			// Distributed as with Phis
			std::get<5>(guess) = l2ball_epsilon();                          // epsilon
			std::get<6>(guess) = l1_proximal_weights();						// l1_proximal_weights
		}
		for (int l = 0; l < target().size(); ++l){
			// Distributed v
			std::get<2>(guess)[l] = t_Vector::Zero(target()[l].size()); // v
			// target needs to be distributed as with Phis
			std::get<4>(guess)[l] = -target()[l]; 						// residual
		}
		return guess;
	}


	protected:
	//! Vector of measurements
	std::vector<t_Vector> target_;
	Vector<Real> total_epsilons;
	psi::mpi::Decomposition decomp_;

	// Records data structures that need to be communicated but don't change throughout the simulation
	bool indices_calculated = false;
	std::vector<std::vector<Vector<t_int>>> indices;
	std::vector<std::vector<Vector<t_int>>> global_indices;


	void iteration_step(t_Vector &out, std::vector<t_Vector> &residual, t_Vector &u, std::vector<t_Vector> &v, t_Vector &x_bar);

	//! Checks input makes sense
	void sanity_check(t_Vector const &x_guess, t_Vector const &u_guess, std::vector<t_Vector> const &v_guess, t_Vector const &x_bar_guess, std::vector<t_Vector> const &res_guess, Vector<Real> const &l2ball_epsilon_guess) const {
		if((!decomp().parallel_mpi() or decomp().global_comm().is_root()) and (Phi()[0]->adjoint() * target()[0]).size() != x_guess.rows())
			PSI_THROW("target, adjoint measurement operator and input vector have inconsistent sizes");
		if(Phi().size() != target().size())
			PSI_THROW("target and vector of measurement operators have inconsistent sizes");
		if((!decomp().parallel_mpi() or decomp().global_comm().is_root()) and (x_bar_guess.size() != x_guess.size()))
			PSI_THROW("x_bar and x have inconsistent sizes");
		if(target().size() != res_guess.size())
			PSI_THROW("target and residual vector have inconsistent sizes");
		if((!decomp().parallel_mpi() or decomp().global_comm().is_root()) and (u_guess.size() != x_guess.size()*levels()))
			PSI_THROW("input vector, measurement operator and dual variable u have inconsistent sizes");
		if(v_guess.size() != target().size())
			PSI_THROW("target and dual variable v have inconsistent sizes");
		if(not static_cast<bool>(is_converged()))
			PSI_WARN("No convergence function was provided: algorithm will run for {} steps", itermax());
		if(l2ball_epsilon_guess.size() != target().size())
			PSI_THROW("target and l2ball_epsilon have inconsistent sizes");
		if(preconditioning()){
			if(target().size() != Ui().size())
				PSI_THROW("target and preconditioning vector have inconsistent sizes");
			for(int i = 0; i<Ui().size();i++){
				if((Ui()[i].array() <= 0.).any())
					PSI_THROW("inconsistent values in the preconditioning vector (each entry must be positive)");
			}
		}
	}

	Diagnostic operator()(t_Vector &out, t_Vector const &x_guess, t_Vector &u_guess, std::vector<t_Vector> &v_guess, t_Vector &x_bar_guess, std::vector<t_Vector> const &res_guess, Vector<Real> &l2ball_epsilon_guess, Vector<Real> &l1_proximal_weights);

};


template <class SCALAR>
void PrimalDualTimeBlocking<SCALAR>::iteration_step(t_Vector &out, std::vector<t_Vector> &residual, t_Vector &u, std::vector<t_Vector> &v, t_Vector &x_bar) {

	// Assuming the root process is a wavelet process
	// Image size, only needed on the root process
	t_Vector prev_sol;

	if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
		prev_sol = out;
	}

	// x_bar should be on the wavelet process(es)
	if(!decomp().parallel_mpi() or decomp().my_number_of_root_wavelets() != 0){
		x_bar =  decomp().my_root_wavelet_comm().broadcast(x_bar, decomp().global_comm().root_id());
	}

	Matrix<t_complex> x_hat;

	if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
		// reshape x_bar
		// Image size. Compute x_hat locally on root.
		auto const image_bar = Image<t_complex>::Map(x_bar.data(), Phi()[0]->imsizey(), Phi()[0]->imsizex());
		// Distributed x_hat to each work after this is calculated.
		// Assume Phi()[0] is on the master.
		// Can just copy x_hat everywhere but don't need to.
		// Could just extract required portions and distribute.
		// The optimisation requires finding the start and end columns in the G matrix for a specific Phi and that
		// would give the start and end indices for the x_hat data to be sent to each process for their Phi. This would
		// then have to be added into a full x_hat vector on each process, but the rest of the x_hat contents can be
		// garbage/null/zero.
		x_hat = Phi()[0]->FFT(image_bar);

	}

	// Calculate the fourier indices for each Phi and then collect them on the frequency root
	// to allow only the specific x_hat and out_hat each process requires to be broadcast to them.
	// This is only done once, because the indices never change throughout a given simulation.
	if(not indices_calculated){
		indices = std::vector<std::vector<Vector<t_int>>>(1);
		indices[0] = std::vector<Vector<t_int>>(decomp().my_frequencies()[0].number_of_time_blocks);
		for(int i=0;i<decomp().my_frequencies()[0].number_of_time_blocks;i++){
			indices[0][i] = Phi()[i]->get_fourier_indices(); // to be transmitted to the master node
		}

		if(decomp().global_comm().is_root()){
			global_indices = std::vector<std::vector<Vector<t_int>>>(1);
			global_indices[0] = std::vector<Vector<t_int>>(decomp().frequencies()[0].number_of_time_blocks);
		}

		decomp().template collect_indices<Vector<t_int>>(indices, global_indices, true);
		indices_calculated = true;

	}

	std::vector<Eigen::SparseMatrix<t_complex>> x_hat_sparse(decomp().my_frequencies()[0].number_of_time_blocks);
	std::vector<Eigen::SparseMatrix<t_complex>> x_hat_global;
	int shapes[decomp().frequencies()[0].number_of_time_blocks][3];
	if(decomp().global_comm().is_root()){
		decomp_.initialise_requests(0,decomp().frequencies()[0].number_of_time_blocks*4);
	}

	if(not decomp().global_comm().is_root()){
		decomp().template receive_fourier_data<t_complex>(x_hat_sparse, 0, 0, true);
	}

	if(decomp().global_comm().is_root()){
		x_hat_global = std::vector<Eigen::SparseMatrix<t_complex>>(decomp().frequencies()[0].number_of_time_blocks);
		auto oversample_factor = Phi()[0]->oversample_factor();
		auto imsizey = Phi()[0]->imsizey();
		auto imsizex = Phi()[0]->imsizex();
#pragma omp parallel for default(shared)
		for(int i=0;i<decomp().frequencies()[0].number_of_time_blocks;i++){
			x_hat_global[i] = Eigen::SparseMatrix<t_complex>(oversample_factor * imsizey * oversample_factor * imsizex, 1);
			x_hat_global[i].reserve(global_indices[0][i].rows());
			for (int k=0; k<global_indices[0][i].rows(); k++){
				x_hat_global[i].insert(global_indices[0][i](k),0) = x_hat(global_indices[0][i](k),0);
			}
			x_hat_global[i].makeCompressed();
		}
		int my_index = 0;
		bool used_this_freq = false;
		for(int i=0;i<decomp().frequencies()[0].number_of_time_blocks;i++){
			decomp_.template send_fourier_data<t_complex>(x_hat_sparse, x_hat_global, &shapes[i][0], i, my_index, 0, used_this_freq, true);
		}
	}

	if(decomp().global_comm().is_root()){
		decomp_.wait_on_requests(0, decomp().frequencies()[0].number_of_time_blocks*4);
		decomp_.cleanup_requests(0);
	}

	// Git_v required on each process, but only need as many as the number of Phi's a process has.
	std::vector<t_Vector> Git_v(Phi().size());

	// v needs to have been distributed here already.
	// x_hat needs too have been distributed here already.
	// target needs to have been distributed already here
	for(int i=0;i<decomp().my_frequencies()[0].number_of_time_blocks;i++){
		t_Vector temp;
		t_Vector v_prox;
		if(preconditioning()){
			temp = v[i] + (Phi()[i]->G_function(x_hat_sparse[i])).eval().cwiseProduct(Ui()[i]);
			ForwardBackward<Scalar> ellipsoid_prox =  ForwardBackward<Scalar>(
					temp.cwiseQuotient(Ui()[i]).eval(),target()[i])
								.Ui(Ui()[i])
								.itermax(itermax_fb())
								.l2ball_epsilon(l2ball_epsilon_(i))
								.relative_variation(relative_variation_fb())
								.decomp(decomp());
			v_prox = ellipsoid_prox().x;
			v[i] = temp - v_prox.cwiseProduct(Ui()[i]);
		}else{
			temp = v[i] + (Phi()[i]->G_function(x_hat_sparse[i])).eval();
			auto const l2ball_proximal = proximal::L2Ball<Scalar>(l2ball_epsilon_(i));
			v_prox = l2ball_proximal(0, temp-target()[i]) + target()[i];
			v[i] = temp - v_prox;
		}
		Git_v[i] = (Phi()[i]->G_function_adjoint(v[i])).eval();
	}

	//Needed to aggregate worker contributions
	t_Vector Phit_v(x_bar.size());

	// This is aggregating the data back to the root, but the first part (the inverse FFT) can be done on the workers.
	// TODO: Check the relative sizes of v1 and v_tmp to see which  is sensible, calculate locally and return the result or
	// just return the data.
	for(int i=0;i<decomp().my_frequencies()[0].number_of_time_blocks;i++){
		// This could be done locally or on the root
		Matrix<t_complex> v1 = Matrix<t_complex>::Map(Git_v[i].data(),Phi()[i]->oversample_factor()*Phi()[i]->imsizey(), Phi()[i]->oversample_factor()*Phi()[i]->imsizex());
		// This could be done locally or on the root
		Image<t_complex> im_tmp = Phi()[i]->inverse_FFT(v1);
		// im_tmp needs to be on the root process now.
		t_Vector v_tmp = t_Vector::Map(im_tmp.data(), im_tmp.size(), 1);
		// On the first iteration initialise Phit_v to v_tmp, subsequently accumulate v_tmp. This
		// stops us having to initialise Phit_v to zero when creating the vector.
		if(i == 0){
			Phit_v = v_tmp;
		}else{
			Phit_v = Phit_v + v_tmp;
		}
	}

	// NEED TO REDUCE Phit_v to root here.
	if(decomp().parallel_mpi()){
		decomp().my_frequencies()[0].freq_comm.distributed_sum(Phit_v,decomp().global_comm().root_id());
	}

	Matrix<t_complex> out_hat;

	if(!decomp().parallel_mpi() or decomp().my_number_of_root_wavelets() != 0){
		// Done on the wavelet processes only.
		// When wavelet parallelised each process needs a copy of x_bar and their portions of Psi and u
		// u_t = u_t-1 + Psi_dagger * x_bar_t-1 - l1norm_prox(u_t-1 + Psi_dagger * x_bar_t-1)
		t_Vector temp2 = u + (Psi().adjoint() * x_bar).eval();
		t_Vector u_prox;
		// Can be done in parallel by each wavelet process
		proximal::l1_norm(u_prox, kappa()*l1_proximal_weights()/sigma1(), temp2);
		// Can be done in parallel by each wavelet process
		u = temp2 - u_prox;

		// This can be done in parallel by each wavelet process
		auto Psi_u = (Psi()(u*sigma1())).eval();

		// Psi_u now needs to be communicated to the root process
		// COMMUNICATION
		if(!decomp().parallel_mpi() or decomp().my_number_of_root_wavelets() != 0){
			decomp().my_root_wavelet_comm().distributed_sum(Psi_u,decomp().global_comm().root_id());
		}

		if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){

			//x_t = positive orth projection(x_t-1 - tau * (sigma1 * Psi * u + sigma2 * Phi dagger * v))
			// out = prev_sol - tau()*(Psi()*u*sigma1() + Phit_v*sigma2());
			// Only done on the root process
			out = prev_sol - tau()*(Psi_u + Phit_v*sigma2());
			if(positivity_constraint()){
				// Only done on the root process
				out = psi::positive_quadrant(out);
			}


			// Only done on the root process
			x_bar = 2*out - prev_sol;
			// x_bar would be need to be redistributed to wavelet processes if the wavelet is being done in parallel

			// out only on the root process
			auto const image_out = Image<t_complex>::Map(out.data(), Phi()[0]->imsizey(), Phi()[0]->imsizex());
			// only done by the root process

			out_hat = Phi()[0]->FFT(image_out);
		}
	}

	std::vector<Eigen::SparseMatrix<t_complex>> out_hat_sparse(decomp().my_frequencies()[0].number_of_time_blocks);
	std::vector<Eigen::SparseMatrix<t_complex>> out_hat_global;
	if(decomp().global_comm().is_root()){
		decomp_.initialise_requests(0,decomp().frequencies()[0].number_of_time_blocks*4);
	}
	if(not decomp().global_comm().is_root()){
		decomp().template receive_fourier_data<t_complex>(out_hat_sparse, 0, 0, true);
	}
	if(decomp().global_comm().is_root()){
		out_hat_global = std::vector<Eigen::SparseMatrix<t_complex>>(decomp().frequencies()[0].number_of_time_blocks);
		auto oversample_factor = Phi()[0]->oversample_factor();
		auto imsizey = Phi()[0]->imsizey();
		auto imsizex = Phi()[0]->imsizex();
#pragma omp parallel for default(shared)
		for(int i=0;i<decomp().frequencies()[0].number_of_time_blocks;i++){
			out_hat_global[i] = Eigen::SparseMatrix<t_complex>(oversample_factor * imsizey * oversample_factor * imsizex, 1);
			out_hat_global[i].reserve(global_indices[0][i].rows());
			for (int k=0; k<global_indices[0][i].rows(); k++){
				out_hat_global[i].insert(global_indices[0][i](k),0) = out_hat(global_indices[0][i](k),0);
			}
			out_hat_global[i].makeCompressed();
		}
		int my_index = 0;
		bool used_this_freq = false;
		for(int i=0;i<decomp().frequencies()[0].number_of_time_blocks;i++){
			decomp_.template send_fourier_data<t_complex>(out_hat_sparse, out_hat_global, &shapes[i][0], i, my_index, 0, used_this_freq, true);
		}
	}

	if(decomp().global_comm().is_root()){
		decomp_.wait_on_requests(0, decomp().frequencies()[0].number_of_time_blocks*4);
		decomp_.cleanup_requests(0);
	}

	//out_hat needs to be sent to all measurement operator processes
	// This is done in parallel
	// residuals are therefore distributed
	for(int i=0; i<decomp().my_frequencies()[0].number_of_time_blocks; i++){
		residual[i] = (Phi()[i]->G_function(out_hat_sparse[i])).eval() - target()[i];
	}

}

template <class SCALAR>
typename PrimalDualTimeBlocking<SCALAR>::Diagnostic PrimalDualTimeBlocking<SCALAR>::
operator()(t_Vector &out, t_Vector const &x_guess, t_Vector &u_guess, std::vector<t_Vector> &v_guess, t_Vector &x_bar_guess, std::vector<t_Vector> const &res_guess, Vector<Real> &l2ball_epsilon_guess, Vector<Real> &l1_proximal_weights_guess) {

	// Functionality to check objective convergence is implemented but not generally required so disabled by default
	bool full_convergence_test = false;

	if(!decomp().parallel_mpi() || decomp().global_comm().is_root()){
		PSI_HIGH_LOG("Performing Primal Dual");
	}

	sanity_check(x_guess, u_guess, v_guess, x_bar_guess, res_guess, l2ball_epsilon_guess);

	std::vector<t_Vector> residual = res_guess;

	// Check if there is a user provided convergence function
	bool const has_user_convergence = static_cast<bool>(is_converged());
	bool converged = false;

	out = x_guess;
	l2ball_epsilon_ = l2ball_epsilon_guess;
	l1_proximal_weights_ = l1_proximal_weights_guess;
	t_uint niters(0);

	decomp().template collect_epsilons<Vector<Real>>(l2ball_epsilon_, total_epsilons);

	std::pair<Real, Real> objectives;

	if(full_convergence_test){
		if(!decomp().parallel_mpi() or decomp().my_number_of_root_wavelets() != 0){
			if(decomp().my_root_wavelet_comm().size() != 1){
				out = decomp().my_root_wavelet_comm().broadcast(out, decomp().global_comm().root_id());
			}
			Real temp_objective = psi::l1_norm(Psi().adjoint() * out, l1_proximal_weights());
			if(decomp().parallel_mpi() and decomp().my_root_wavelet_comm().size() != 1){
				decomp().my_root_wavelet_comm().distributed_sum(&temp_objective,decomp().global_comm().root_id());
			}
			if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
				objectives = {temp_objective, 0.};
			}
		}
	}

	Vector<int> counter =  Vector<int>::Zero(decomp().my_frequencies()[0].number_of_time_blocks);

	for(; niters < itermax(); ++niters) {

		t_Vector x_old;
		// Currently x_old is only on the root process as with out
		if(!decomp().parallel_mpi() || decomp().global_comm().is_root()){
			x_old = out;
		}

		if(!decomp().parallel_mpi() || decomp().global_comm().is_root()){
			PSI_HIGH_LOG("    - Iteration {}/{}", niters, itermax());
		}
		iteration_step(out, residual, u_guess, v_guess, x_bar_guess);

		Vector<Vector<Real>> residual_norms = Vector<Vector<Real>>(1);
		residual_norms[0] = Vector<Real>(decomp().my_frequencies()[0].number_of_time_blocks);

		for (int j = 0; j < target().size(); ++j) {
			residual_norms[0][j] = residual[j].stableNorm();
		}

		Vector<Vector<Real>> total_residual_norms = Vector<Vector<Real>>(decomp().global_number_of_frequencies());
		for(int f=0; f<decomp().global_number_of_frequencies(); f++){
			total_residual_norms[f] = Vector<Real>(decomp().frequencies()[f].number_of_time_blocks);
		}

		decomp().template collect_residual_norms<Real>(residual_norms, total_residual_norms);

		if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
			// Done in parallel as with the phi

			// Need to do a sum reduce here to get the residual norm back to the root
			// COMMUNICATION
			// Only call this on the root
			PSI_HIGH_LOG("      - Sum of residuals: {}", total_residual_norms[0].sum());

		}
		if(full_convergence_test){
			objectives.second = objectives.first;

			// Could be parallelised with the wavelet processes but that would require out to be distributed here
			if(!decomp().parallel_mpi() or decomp().my_number_of_root_wavelets() != 0){
				if(decomp().my_root_wavelet_comm().size() != 1){
					out = decomp().my_root_wavelet_comm().broadcast(out, decomp().global_comm().root_id());
				}
				Real temp_objective = psi::l1_norm(Psi().adjoint() * out, l1_proximal_weights());
				if(decomp().parallel_mpi() and decomp().my_root_wavelet_comm().size() != 1){
					decomp().my_root_wavelet_comm().distributed_sum(&temp_objective,decomp().global_comm().root_id());
				}
				if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
					objectives.first = temp_objective;
				}
			}
		}

		bool rel_x_check;

		if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){

			// update epsilon
			// Only done on the root process
			auto const rel_x = (out - x_old).norm();
			rel_x_check = (rel_x < relative_variation_x()*out.norm());

		}

		rel_x_check = decomp().my_frequencies()[0].freq_comm.broadcast(rel_x_check, decomp().global_comm().root_id());

		if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){

			t_real relative_objective;
			if(full_convergence_test){
				relative_objective = std::abs(objectives.first - objectives.second) / objectives.first;
				PSI_HIGH_LOG("    - objective: obj value = {}, rel obj = {}", objectives.first,
						relative_objective);
			}
			Vector<Real> total_residual_convergence = residual_convergence() * total_epsilons;

			PSI_HIGH_LOG("      - residual norm = {}, residual convergence = {}", total_residual_norms[0].norm(), total_residual_convergence.norm());

			auto const user = (not has_user_convergence) or is_converged(out);
			auto const res = (residual_convergence() <= 0e0) or (total_residual_norms[0].norm() < total_residual_convergence.norm());
			bool rel;
			if(full_convergence_test){
				rel = relative_variation() <= 0e0 or relative_objective < relative_variation();
			}
			// Only done on the root
			if(full_convergence_test){
				converged = user and res and rel;
			}else{
				converged = user and res and rel_x_check;
			}
		}


		if(decomp().parallel_mpi()){
			int temp_converged;
			// Setting the value to be broadcast. This is done on all processes
			// but the value only matters on the root process
			if(converged){
				temp_converged = 1;
			}else{
				temp_converged = 0;
			}
			temp_converged = decomp().my_frequencies()[0].freq_comm.broadcast(temp_converged, decomp().global_comm().root_id());
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
			psi::io::IOStatus checkpoint_status = check.checkpoint_time_blocking_with_collect(decomp(), filename, out, l2ball_epsilon(), l1_proximal_weights(), kappa(), sigma2(), image_size(), delta(), current_reweighting_iter());
			if((!decomp().parallel_mpi() or decomp().global_comm().is_root()) and checkpoint_status != psi::io::IOStatus::Success){
				PSI_HIGH_LOG("Problem checkpointing to file, error is: {}", psi::io::IO<Scalar>::GetErrorMessage(checkpoint_status));
			}
		}

		// Only print on the root
		if(converged){
			if(!decomp().parallel_mpi() or decomp().global_comm().is_root()){
				PSI_HIGH_LOG("    - converged in {} of {} iterations", niters, itermax());
			}
			break;
		}

		if(update_epsilon() && niters > adaptive_epsilon_start() && rel_x_check){
			// Done in parallel as with the phi

			bool eps_updated = false;
			for(int l = 0; l < decomp().my_frequencies()[0].number_of_time_blocks; ++l){
				bool eps_flag = residual_norms[0][l] < lambdas()(0) * l2ball_epsilon_(l) or
						residual_norms[0][l] > lambdas()(1) * l2ball_epsilon_(l);
				if(niters > counter(l) + P() and eps_flag){

					l2ball_epsilon_(l) =
							lambdas()(2) * residual_norms[0][l] + (1 - lambdas()(2)) * l2ball_epsilon_(l);
					PSI_HIGH_LOG("{}: epsilon updated: {}", decomp().global_comm().rank(), l2ball_epsilon_(l));
					counter(l) = niters;
					eps_updated = true;
				}
			}
			decomp().template collect_epsilons<Vector<Real>>(l2ball_epsilon(), total_epsilons);
			if(eps_updated and (!decomp().parallel_mpi() or decomp().global_comm().is_root())){
				PSI_HIGH_LOG("epsilon norm: {}",total_epsilons.norm());
				PSI_HIGH_LOG("epsilon l2 norm: {}",total_epsilons.stableNorm());
			}
		}

	}

	// check function exists, otherwise, don't know if convergence is meaningful
	// Only do on root process
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
