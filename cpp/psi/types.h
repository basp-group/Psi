#ifndef BICO_TRAITS_H
#define BICO_TRAITS_H

#include "psi/config.h"
#include <complex>
#include <vector>
#include <functional>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "psi/real_type.h"

namespace psi {

//! Root of the type hierarchy for signed integers
typedef int t_int;
//! Root of the type hierarchy for unsigned integers
typedef size_t t_uint;
//! Root of the type hierarchy for real numbers
typedef double t_real;
//! Root of the type hierarchy for (real) complex numbers
typedef std::complex<t_real> t_complex;

//! \brief A vector of a given type
//! \details Operates as mathematical vector.
template <class T = t_real> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <class T = t_real> using VectorBlock = Eigen::VectorBlock<T, Eigen::Dynamic>;


//! \brief A matrix of a given type
//! \details Operates as mathematical matrix.
//template <class T = t_real> using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <class T = t_real> using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <class T = t_real> using RowMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
//! \brief A 1-dimensional list of elements of given type
//! \details Operates coefficient-wise, not matrix-vector-wise
template <class T = t_real> using Array = Eigen::Array<T, Eigen::Dynamic, 1>;
//! \brief A 2-dimensional list of elements of given type
//! \details Operates coefficient-wise, not matrix-vector-wise
//template <class T = t_real> using Image = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <class T = t_real> using Image = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;

//! Typical function out = A*x
template <class VECTOR = Vector<>>
using OperatorFunction = std::function<void(VECTOR &, VECTOR const &)>;
template <class MATRIX = Matrix<>, class IMAGE = Image<>>
using MatrixImageFunction = std::function<void(MATRIX &, IMAGE const &)>;
template <class IMAGE = Image<>, class MATRIX = Matrix<>>
using ImageMatrixFunction = std::function<void(IMAGE &, MATRIX const &)>;
template <class VECTOR = Vector<>, class MATRIX = Matrix<>>
using VectorMatrixFunction = std::function<void(VECTOR &, MATRIX const &)>;

//! Typical function signature for calls to proximal
template <class SCALAR = t_real>
using ProximalFunction = std::function<void(Vector<SCALAR> &, typename real_type<SCALAR>::type,
                                            Vector<SCALAR> const &)>;
//! Typical function signature for convergence
template <class SCALAR = t_real>
using ConvergenceFunction = std::function<bool(Vector<SCALAR> const &)>;
template <class SCALAR = t_real>
using ConvergenceMatrixFunction = std::function<bool(Matrix<SCALAR> const &)>;

//! Additional type to easily create a MATLAB cell-like structure (containing matrices of the same type)
namespace is_eigen_matrix_detail {
    // These functions are never defined.
    template <typename T>
    std::true_type test(const Eigen::MatrixBase<T>*);
    std::false_type test(...);
}

template <typename T>
struct is_eigen_matrix
    : public decltype(is_eigen_matrix_detail::test(std::declval<T*>()))
{};

template <typename T, typename Enable = void>
using Cell = std::vector<T>;

template <typename Derived>
using EigenCell = Cell<Derived, typename std::enable_if<is_eigen_matrix<Derived>::value>::type>;

}
#endif
