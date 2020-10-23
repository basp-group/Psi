#ifndef PSI_MPI_SCALAPACK_H
#define PSI_MPI_SCALAPACK_H

#include "psi/config.h"

#ifdef PSI_SCALAPACK
#include <memory>
#ifdef PSI_SCALAPACK_MKL
#include <mkl_blacs.h>
#include <mkl.h>
#endif
#include <set>
#include <string>
#include <type_traits>
#include <vector>
#include <Eigen/Core>
#include "psi/types.h"
#include "psi/mpi/communicator.h"
#include "psi/mpi/decomposition.h"
#include "psi/logging.h"

extern "C"
{
void descinit_(int *, int *, int *, int *, int *,
		int *, int *, int *, int *, int *);
int numroc_(int *, int *, int *, int *, int *);
int indxl2g_(int *, int *, int *, int *, int *);

void pdgesvd_(char*, char*, int*, int*,double*,int*,int*,int*,double*,
		double*,int*,int*,int*,
		double*,int*,int*,int*,
		double*,int*,int*);
void pzgesvd_(char*, char*, int*, int*,double*,int*,int*,int*,double*,
		double*,int*,int*,int*,
		double*,int*,int*,int*,
		double*,int*,double*,int*);

#ifndef PSI_SCALAPACK_MKL
void blacs_pinfo_(int *, int *);

void blacs_abort_(int *, int *);

void  blacs_get_(int *, int *, int *);

void  blacs_gridinit_(int *, char *, int *, int *);

void  blacs_gridinfo_(int *, int *, int *, int *, int *);

void  blacs_barrier_(int *, char *);

void  blacs_gridexit_(int *);

void  blacs_exit_(int *);

void  dgesd2d_(int *, int *, int *, double *, int *, int *, int *);

void  dgerv2d_(int *, int *, int *, double *, int *, int *, int *);
#endif
}



namespace psi {
namespace mpi {

//! \brief Setup the blacs decomposition and the send recv functionality for the scalapack svd
//!
class Scalapack {

public:

	//! Constructor
	Scalapack(bool using_scalapack);

	~Scalapack(){
		// Calling blacs_gridexit_ actually calls MPI Abort, which we don't really want if we're just
		// cleaning up an object
		if(involved_process_group_ != i_negone_){
		//	blacs_gridexit_(&involved_process_group_);
		}
		if(all_processes_ != i_negone_){
		//	blacs_gridexit_(&all_processes_);
		}
	};

	bool involvedInSVD();
	bool setupBlacs(Decomposition decomp, int number_of_processes, int M, int N);
	std::pair<int, int> process_grid_size(int number_of_processes);
	bool setupSVD(Vector<t_real> local_Data, Vector<t_real> &output);
	bool setupSVD(Vector<t_real> local_Data, Vector<t_real> &output, Vector<t_real> &output_u, Vector<t_real> &output_vt);
	bool runSVD(Vector<t_real> local_Data, Vector<t_real> &output);
	bool runSVD(Vector<t_real> local_Data, Vector<t_real> &output,  Vector<t_real> &output_u, Vector<t_real> &output_vt);
	bool scatter(Decomposition decomp, Vector<t_real> &local_data, Matrix<t_real> total_data, int M, int N, int mp, int np);
	bool scatter(Decomposition decomp, Vector<t_real> &local_data, Vector<t_real> total_data, int M, int N, int mp, int np);
	bool gather(Decomposition decomp, Vector<t_real> local_data, Vector<t_real> &total_data, int M, int N, int mp, int np);
	bool sendToScalapackRoot(Decomposition decomp, Vector<t_real> &data_svd);
	bool recvFromScalapackRoot(Decomposition decomp, Vector<t_real> &data_svd);

	bool usingScalapack() { return using_scalapack_; }
	int getM() const { return M_; }
	int getN() const { return N_; }
	int getnprow() const { return nprow_; }
	int getnpcol() const { return npcol_; }
	int getmb() const { return mb_; }
	int getnb() const { return nb_; }
	int getmpa() const { return mpa_; }
	int getnpa() const { return npa_; }
	int getmpu() const { return mpu_; }
	int getnpu() const { return npu_; }
	int getmpvt() const { return mpvt_; }
	int getnpvt() const { return npvt_; }
	int getmyrow() const { return myrow_; }
	int getmycol() const { return mycol_; }
	int getllda() const { return llda_; }
	int getlldu() const { return lldu_; }
	int getlldvt() const { return lldvt_; }
	int* getdesca() { return desca_; }
	int* getdescu() { return descu_; }
	int* getdescvt() { return descvt_; }
	int getNumberOfProcesses() const { return number_of_processes_; }
	bool getBlacsState() const { return setup_; }
	bool getSVDState() const { return work_setup_; }
	bool getSetupState() const { return setup_ and work_setup_; }
	Communicator scalapack_comm() const { return scalapack_comm_; }


protected:

	// Class data
//	psi::mpi::Decomposition decomp_;
	bool using_scalapack_ = false;
	int number_of_processes_ = -1;
	int i_negone_ = -1;
	int i_one_  = 1;
	int i_zero_ = 0;
	int initial_process_handle_ = i_negone_; // Initial blacs process handle
	int involved_process_group_ = i_negone_; // Processes involved in the SVD and other scalapack operations
	int all_processes_ = i_negone_; // All MPI processes
	int minMN_;
	int M_;
	int N_;
	int nprow_;
	int npcol_;
	int myrow_;
	int mycol_;
	int mb_;
	int nb_;
	int mpa_;
	int npa_;
	int mpu_;
	int npu_;
	int mpvt_;
	int npvt_;
	int llda_;
	int lldu_;
	int lldvt_;
	int info_ = 0;
	int lwork_;
	std::vector<double> work_;
	std::vector<double> rwork_;
	int desca_[9];
	int descu_[9];
	int descvt_[9];
	// setup_ is used to record whether the blacs library has been setup for this scalapack object
	// This needs to be done before any of the gather, scatter, or SVD routines are run.
	bool setup_ = false;
	// work_setup_ is used to record whether the work arrays have been setup for the SVD.
	// The scalapack SVD requires work arrays of the correct size to be setup before the SVD is
	// run. These can be calculated by the SVD routine itself if run in the right way, and that
	// functionality has been implemented in the setupSVD routine. Once setupSVD is run successfully,
	// this variable should be set to true and the runSVD routine can be used.
	bool work_setup_ = false;
	// Communicator for the scalapack workers
	Communicator scalapack_comm_ = Communicator::None();

private:



};

// Return true if a process is in the blacs involved group (i.e. if it's rank is below
// number_of_processes passed to setupBlacs) and false otherwise. Enables selecting only
// those processes involved in the SVD work externally to this class.
bool Scalapack::involvedInSVD(){
	return involved_process_group_ >= 0;
}

} // namespace mpi
} // namespace psi
#endif /* ifdef PSI_SCALAPACK */
#endif /* ifndef PSI_MPI_SCALAPACK_H */
