#include <sstream>
#include <benchmark/benchmark.h>
#include <psi/l1_proximal.h>
#include <psi/real_type.h>
#include <psi/types.h>

template <class TYPE> void function_l1p(benchmark::State &state) {
  typedef typename psi::real_type<TYPE>::type Real;
  auto const N = state.range(0);
  auto const input = psi::Vector<TYPE>::Random(N).eval();
  auto const Psi = psi::Matrix<TYPE>::Random(input.size(), input.size() * 10).eval();
  psi::Vector<Real> const weights
      = psi::Vector<TYPE>::Random(Psi.cols()).normalized().array().abs();

  auto const l1 = psi::proximal::L1<TYPE>()
                      .tolerance(std::pow(10, -state.range(1)))
                      .itermax(100)
                      .fista_mixing(true)
                      .positivity_constraint(true)
                      .nu(1)
                      .Psi(Psi)
                      .weights(weights);

  Real const gamma = 1e-2 / Psi.array().abs().sum();
  auto output = psi::Vector<TYPE>::Zero(N).eval();
  while(state.KeepRunning())
    l1(output, gamma, input);
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(N) * sizeof(TYPE));
}

BENCHMARK_TEMPLATE(function_l1p, psi::t_complex)->RangePair(1, 256, 4, 12)->UseRealTime();
BENCHMARK_TEMPLATE(function_l1p, psi::t_real)->RangePair(1, 256, 4, 12)->UseRealTime();

BENCHMARK_MAIN();
