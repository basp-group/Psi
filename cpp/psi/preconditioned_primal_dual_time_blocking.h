#ifndef PSI_PRECONDITIONED_PRIMAL_DUAL_TIME_BLOCKING_H
#define PSI_PRECONDITIONED_PRIMAL_DUAL_TIME_BLOCKING_H

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
#include "psi/primal_dual_time_blocking.h"

namespace psi {
namespace algorithm {

//! \brief Primal Dual method
//! \details This is a basic implementation of the primal dual method using a forward backwards algorithm.
//! \f$\min_{, y, z} f() + l(y) + h(z)\f$ subject to \f$Φx = y\f$, \f$Ψ^Hx = z\f$
template <class SCALAR> class PreconditionedPrimalDualTimeBlocking : public PrimalDualTimeBlocking<SCALAR> {
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


	//! Setups PreconditionedPrimalDualTimeBlocking
	//! Setups PreconditionedPrimalDualBlocking
	template <class T>
	PreconditionedPrimalDualTimeBlocking(EigenCell<T> const &target, const t_uint &image_size, const Vector<Real> &l2ball_epsilon, std::vector<std::shared_ptr<const t_LinearTransform>> &Phi, std::vector<Vector<Real>> const &Ui)
	: PrimalDualBlocking<SCALAR>(target, image_size, l2ball_epsilon, Phi), Ui_(Ui), itermax_fb_(20), relative_variation_fb_(1e-8),  {}
	virtual ~PreconditionedPrimalDualBlocking() {}

	// Macro helps define properties that can be initialized as in
	// auto pd  = PreconditionedPrimalDualTimeBlocking<float>().prop0(value).prop1(value);
#define PSI_MACRO(NAME, TYPE)                                                                     \
		TYPE const &NAME() const { return NAME##_; }                                                     \
		PreconidtionedPrimalDualTimeBlocking<SCALAR> &NAME(TYPE const &NAME) {                                                     \
			NAME##_ = NAME;                                                                                \
			return *this;                                                                                  \
		}                                                                                                \
		\
protected:                                                                                         \
TYPE NAME##_;                                                                                    \
\
public:

	//! Preconditioning vector (equivalent to a diagonal preconditioner)
	PSI_MACRO(Ui, std::vector<Vector<Real>>);
	//! Maximum number of inner iterations (projection onto the ellipsoid)
	PSI_MACRO(itermax_fb, t_uint);
	//! Relative variation (stopping criterion)
	PSI_MACRO(relative_variation_fb, Real);

#undef PSI_MACRO


	protected:


	void iteration_step(t_Vector &out, std::vector<t_Vector> &residual, t_Vector &u, std::vector<t_Vector> &v, t_Vector &x_bar);

	//! Checks input makes sense
	void sanity_check(t_Vector const &x_guess, t_Vector const &u_guess, std::vector<t_Vector> const &v_guess, t_Vector const &x_bar_guess, std::vector<t_Vector> const &res_guess, Vector<Real> const &l2ball_epsilon_guess) const {
		PrimalDualBlocking<SCALAR>::sanity_check(x_guess, u_guess, v_guess, x_bar_guess, res_guess, l2ball_epsilon_guess);
		if(PrimalDualBlocking<SCALAR>::target().size() != Ui().size())
			PSI_THROW("target and preconditioning vector have inconsistent sizes");
		if((Ui().array() <= 0.).any())
			PSI_THROW("inconsistent values in the preconditioning vector (each entry must be positive)");
	}

};

template <class SCALAR>
void PreconditionedPrimalDualTimeBlocking<SCALAR>::iteration_step(t_Vector &out, std::vector<t_Vector> &residual, t_Vector &u,
		std::vector<t_Vector> &v, t_Vector &x_bar) {

	// Assuming the root process is a wavelet process
	// Image size, only needed on the root process
	t_Vector prev_sol = out;
	// Levels * Image size. only needed on the wavelet process(es)
	t_Vector prev_u = u;
	// Measurement operator size, one element of the std::vector per process or however the decomposition is done
	// v should be created distributed.
	std::vector<t_Vector> prev_v = v;

	// x_bar should be on the wavelet process(es)

	Matrix<t_complex> x_hat;

	if(!decomp().parallel_mpi() || decomp().global_comm().is_root()){
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

	if(decomp().parallel_mpi()){
		if(x_hat_rows == -1 || x_hat_cols == -1){
			if(decomp().global_comm().is_root()){
				x_hat_rows = x_hat.rows();
				x_hat_cols = x_hat.cols();
			}
			x_hat_rows = decomp().my_frequencies()[0].freq_comm.broadcast(x_hat_rows, decomp().global_comm().root_id());
			x_hat_cols = decomp().my_frequencies()[0].freq_comm.broadcast(x_hat_cols, decomp().global_comm().root_id());
		}
		if(!decomp().global_comm().is_root()){
			x_hat = Matrix<t_complex>(x_hat_rows, x_hat_cols);
		}
		x_hat =  decomp().my_frequencies()[0].freq_comm.broadcast(x_hat, decomp().global_comm().root_id());
	}

	// Git_v required on each process, but only need as many as the number of Phi's a process has.
	std::vector<t_Vector> Git_v(Phi().size());

	// v needs to have been distributed here already.
	// x_hat needs too have been distributed here already.
	// target needs to have been distributed already here
	for(int i=0;i<decomp().my_frequencies()[0].number_of_time_blocks;i++){
		t_Vector temp = v[i] + (Phi()[i]->G_function(x_hat)).eval().cwiseProduct(static_cast<t_Vector>(Ui()[i]));
		t_Vector v_prox;
		algorithm::ForwardBackward<Scalar> ellipsoid_prox = algorithm::ForwardBackward<Scalar>(temp.cwiseQuotient(static_cast<t_Vector>(Ui()[i])), PrimalDualBlocking<SCALAR>::target())
                        		 .Ui(Ui()[i])
								 .itermax(itermax_fb())
								 .l2ball_epsilon(PrimalDualBlocking<SCALAR>::l2ball_epsilon_[i]) // check accessor...
								 .relative_variation(relative_variation_fb());

		v_prox = ellipsoid_prox();
		v[i] = temp - v_prox;
		Git_v[i] = (Phi()[i]->G_function_adjoint(v[i])).eval();

	}


	//Needed to aggregate worker contributions
	t_Vector Phit_v = t_Vector::Zero(out.size());

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
		Phit_v = Phit_v + v_tmp;
	}

	// NEED TO REDUCE Phit_v to root here.
	if(decomp().parallel_mpi()){
		decomp().my_frequencies()[0].freq_comm.distributed_sum(Phit_v,decomp().global_comm().root_id());
	}

	Matrix<t_complex> out_hat;

	// Currently only doing the wavelets on the roota
	if(!decomp().parallel_mpi() || decomp().global_comm().is_root()){
		// Done on the wavelet processes only.
		// If parallelised each process would need a copy of x_bar and their portions of Psi and u
		// u_t = u_t-1 + Psi_dagger * x_bar_t-1 - l1norm_prox(u_t-1 + Psi_dagger * x_bar_t-1)
		t_Vector temp2 = u + (Psi().adjoint() * x_bar).eval();
		t_Vector u_prox;
		// Can be done in parallel by each wavelet process
		proximal::l1_norm(u_prox, kappa()/sigma1(), temp2);
		// Can be done in parallel by each wavelet process
		u = temp2 - u_prox;

		// This can be done in parallel by each wavelet process
		auto Psi_u = (Psi()*u*sigma1()).eval();


		// Psi_u now needs to be communicated to the root process
		// COMMUNICATION

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

	if(decomp().parallel_mpi()){
		if(out_hat_rows == -1 || out_hat_cols == -1){
			if(decomp().global_comm().is_root()){
				out_hat_rows = out_hat.rows();
				out_hat_cols = out_hat.cols();
			}
			out_hat_rows = decomp().my_frequencies()[0].freq_comm.broadcast(out_hat_rows, decomp().global_comm().root_id());
			out_hat_cols = decomp().my_frequencies()[0].freq_comm.broadcast(out_hat_cols, decomp().global_comm().root_id());
		}
		if(!decomp().global_comm().is_root()){
			out_hat = Matrix<t_complex>(out_hat_rows, out_hat_cols);
		}

		out_hat = decomp().my_frequencies()[0].freq_comm.broadcast(out_hat, decomp().global_comm().root_id());
	}

	//out_hat needs to be sent to all measurement operator processes
	// This is done in parallel
	// residuals are therefore distributed
	for(int i=0; i<decomp().my_frequencies()[0].number_of_time_blocks; i++){
		residual[i] = (Phi()[i]->G_function(out_hat)).eval() - target()[i];
	}

} /* end of preconditioned primal-dual */
} /* psi::algorithm */
} /* psi */
#endif
