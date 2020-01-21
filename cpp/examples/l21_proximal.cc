#include <psi/maths.h>
#include <psi/proximal.h>
#include <psi/types.h>
#include <iostream>
#include <math.h>

using namespace std;

int main(int args, char const ** argv) {

  typedef psi::t_real Scalar;
  typedef psi::Matrix<Scalar> Matrix;
  typedef psi::Array<Scalar> Array;
  typedef psi::Vector<Scalar> Vector;

  int Nx=5, Ny=3;

  Matrix x=Matrix::Random(Nx,Ny);
  Vector w=Vector::Random(Nx);
  w = w.cwiseAbs();
  Vector th = Vector::Random(Nx);
  Matrix outx(Nx,Ny), l21prox(Nx,Ny);

//  x << 1.76405235,  0.40015721,  0.97873798,
//	2.2408932 ,  1.86755799, -0.97727788,
//	0.95008842, -0.15135721, -0.10321885,
//	0.4105985 ,  0.14404357,  1.45427351,
//	0.76103773,  0.12167502,  0.44386323;

  auto out = psi::l21_norm(x, w);

  Scalar l21norm = 0;

  for(int i=0;i<Nx;i++)
  {
	  Scalar tmp = 0;
	  for(int j=0;j<Ny;j++)
		  tmp += x(i,j)*x(i,j);
	  l21norm += sqrt(tmp)* w(i);
  }

  if(std::abs(out - l21norm) > 1e-15)
      throw std::exception();

  psi::proximal::l21_norm(outx, x, w);

  for(int i=0;i<Nx;i++)
  {
  	  Scalar tmp = 0;
  	  for(int j=0;j<Ny;j++)
  		  tmp += x(i,j) * x(i,j);
  	  tmp = sqrt(tmp);
  	  Scalar coef = fmax(tmp-w(i),0);
  	  for(int j=0;j<Ny;j++)
		    l21prox(i,j) = x(i,j)/tmp * coef;
  }

  if((outx - l21prox).norm() > 1e-15)
	  throw std::exception();

  return 0;
}




