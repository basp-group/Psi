#include <psi/proximal.h>
#include <psi/types.h>
#include <iostream>
#include <math.h>
#include <Eigen/SVD>

using namespace std;

int main(int args, char const ** argv) {

  typedef psi::t_real Scalar;
  typedef psi::Matrix<Scalar> Matrix;
  typedef psi::Vector<Scalar> Vector;

  int Nx=5, Ny=3;

  Matrix out=Matrix(Nx,Ny), w=Matrix::Ones(Ny,1);
  Vector Diag= Vector::LinSpaced(3, 3, 0);
  Matrix x = Matrix(Diag.asDiagonal());

  psi::proximal::nuclear_norm(out, x, w);

  auto res = Matrix((Diag-w).cwiseMax(0.).asDiagonal());

  if(out != res)
  	  throw std::exception();
  return 0;
}



