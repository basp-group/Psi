#ifndef PSI_TRANSFORM_OPERATORS_H
#define PSI_TRANSFORM_OPERATORS_H

#include "psi/config.h"
#include <array>
#include <memory>
#include <type_traits>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "psi/logging.h"
#include "psi/types.h"
#include "psi/maths.h"
#include "psi/wrapper.h"

namespace psi {

//! Joins together direct and indirect operators
template <class VECTOR> class LinearTransformOperations {
public:



	  // The functions below are stub functions required to enable measurement operators
	  // to call FFT and G matrix functionality independently.
	  virtual Matrix<t_complex> FFT(const Image<t_complex> &eigen_image) const{
		PSI_HIGH_LOG("Wrong FFT routine being called");
	    return eigen_image;
	  }

	  virtual Image<t_complex> inverse_FFT(Matrix<t_complex> &ft_vector) const{
			PSI_HIGH_LOG("Wrong inverse FFT routine being called");

	    return ft_vector;
	  }

	  virtual Vector<t_complex> G_function(const Matrix<t_complex> &ft_vector) const{
			PSI_HIGH_LOG("Wrong G function routine being called");

	    return ft_vector;
	  }

		virtual Vector<t_complex> G_function(const Eigen::SparseMatrix<t_complex> &ft_vector) const{
			PSI_HIGH_LOG("Wrong G function routine being called");

	    return Vector<t_complex>::Ones(0);
	  }

	  virtual Vector<t_complex> G_function_adjoint(const Vector<t_complex> &visibilities) const{
			PSI_HIGH_LOG("Wrong G function adjoint routine being called");
	    return visibilities;
	  }

		virtual Vector<t_int> get_fourier_indices() const{
			PSI_HIGH_LOG("Wrong fourier indices routine being called");
			Vector<t_int> out = Vector<t_int>::Ones(0,0);
	    return out;
	  }

	  virtual void enable_preconditioning() {
		  PSI_HIGH_LOG("Wrong enable_preconditioning function called");
		  return;
	  }

	  virtual void disable_preconditioning() {
		  PSI_HIGH_LOG("Wrong disable_preconditioning set function called");
		  return;
	  }

};

}
#endif
