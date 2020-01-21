#ifndef PSI_MPI_TYPES_H
#define PSI_MPI_TYPES_H

#include "psi/config.h"
#ifdef PSI_MPI
#include <complex>
#include <mpi.h>
namespace psi {
  namespace mpi {
    //! Type of an mpi type
    //typedef decltype(MPI_CHAR) MPIType;
    // Some MPI libraries don't actually have a type defined that will work in the above line
    // so the line below can be used instead
    typedef MPI_Datatype MPIType;
    
    //! MPI type associated with a c++ type
    template <class T> struct Type;
    
    static_assert(not std::is_same<char, std::int8_t>::value, "");
#define PSI_MACRO(TYPE)							\
    template <> struct Type<TYPE> { static const MPIType value; };
    PSI_MACRO(std::int8_t);
    PSI_MACRO(std::int16_t);
    PSI_MACRO(std::int32_t);
    PSI_MACRO(std::int64_t);
    PSI_MACRO(std::uint8_t);
    PSI_MACRO(std::uint16_t);
    PSI_MACRO(std::uint32_t);
    PSI_MACRO(std::uint64_t);
    
#ifndef PSI_CHAR_ARCH
    PSI_MACRO(char);
#endif
#ifndef PSI_LONG_ARCH
    PSI_MACRO(signed long);
#endif
#ifndef PSI_ULONG_ARCH
    PSI_MACRO(unsigned long);
#endif
    
    PSI_MACRO(int *);
    PSI_MACRO(float);
    PSI_MACRO(float *);
    PSI_MACRO(double);
    PSI_MACRO(double *);
    PSI_MACRO(long double);
    PSI_MACRO(long double *);
    PSI_MACRO(std::complex<float>);
    PSI_MACRO(std::complex<float> *);
    PSI_MACRO(std::complex<double>);
    PSI_MACRO(std::complex<double> *);
    PSI_MACRO(std::complex<long double>);
    PSI_MACRO(std::complex<long double> *);
#undef PSI_MACRO


    //! MPI type associated with a c++ type
    template <class T> inline constexpr MPIType registered_type(T const &) { return Type<T>::value; }
    
    namespace details {
      template<typename... Ts> struct make_void { typedef void type;};
      //! \brief Defines c++17 metafunction
      //! \details This implements [std::void_t](http://en.cppreference.com/w/cpp/types/void_t). See
      //! therein and [CWG 1558](http://open-std.org/JTC1/SC22/WG21/docs/cwg_defects.html#1558) for the
      //! reason behind the slightly convoluted approach.
      template<typename... Ts> using void_t = typename make_void<Ts...>::type;
    }
    //! True if the type is registered
    template <class T, class = details::void_t<>> class is_registered_type : public std::false_type {};
    template <class T>
      class is_registered_type<T, details::void_t<decltype(Type<T>::value)>> : public std::true_type {};
  } /* psi::mpi */
} /* psi */
#endif
#endif /* ifndef PSI_TYPES */
