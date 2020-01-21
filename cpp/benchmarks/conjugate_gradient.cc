#include "psi/conjugate_gradient.h"
#include <sstream>
#include <benchmark/benchmark.h>

template <class TYPE> void matrix_cg(benchmark::State &state) {
  auto const N = state.range(0);
  auto const epsilon = std::pow(10, -state.range(1));
  auto const A = psi::Image<TYPE>::Random(N, N).eval();
  auto const b = psi::Array<TYPE>::Random(N).eval();

  auto const AhA = A.matrix().transpose().conjugate() * A.matrix();
  auto const Ahb = A.matrix().transpose().conjugate() * b.matrix();
  auto output = psi::Vector<TYPE>::Zero(N).eval();
  psi::ConjugateGradient cg(0, epsilon);
  while(state.KeepRunning())
    cg(output, AhA, Ahb);
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(N) * sizeof(TYPE));
}

template <class TYPE> void function_cg(benchmark::State &state) {
  auto const N = state.range(0);
  auto const epsilon = std::pow(10, -state.range(1));
  auto const A = psi::Image<TYPE>::Random(N, N).eval();
  auto const b = psi::Array<TYPE>::Random(N).eval();

  auto const AhA = A.matrix().transpose().conjugate() * A.matrix();
  auto const Ahb = A.matrix().transpose().conjugate() * b.matrix();
  typedef psi::Vector<TYPE> t_Vector;
  auto func = [&AhA](t_Vector &out, t_Vector const &input) { out = AhA * input; };
  auto output = psi::Vector<TYPE>::Zero(N).eval();
  psi::ConjugateGradient cg(0, epsilon);
  while(state.KeepRunning())
    cg(output, func, Ahb);
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(N) * sizeof(TYPE));
}

BENCHMARK_TEMPLATE(matrix_cg, psi::t_complex)->RangePair(1, 256, 4, 12)->UseRealTime();
BENCHMARK_TEMPLATE(matrix_cg, psi::t_real)->RangePair(1, 256, 4, 12)->UseRealTime();
BENCHMARK_TEMPLATE(function_cg, psi::t_complex)->RangePair(1, 256, 4, 12)->UseRealTime();
BENCHMARK_TEMPLATE(function_cg, psi::t_real)->RangePair(1, 256, 4, 12)->UseRealTime();

BENCHMARK_MAIN();
