#include <sstream>
#include <benchmark/benchmark.h>
#include <psi/wavelets/wavelets.h>

unsigned get_size(unsigned requested, unsigned levels) {
  auto const N = (1u << levels);
  return requested % N == 0 ? requested : requested + N - requested % N;
}
std::string get_name(unsigned db) {
  std::ostringstream sstr;
  sstr << "DB" << db;
  return sstr.str();
}

template <class TYPE, unsigned DB = 1, unsigned LEVEL = 1>
void direct_matrix(benchmark::State &state) {
  auto const Nx = get_size(state.range(0), LEVEL);
  auto const Ny = get_size(state.range(1), LEVEL);
  auto const input = psi::Image<TYPE>::Random(Nx, Ny).eval();
  auto output = psi::Image<TYPE>::Zero(Nx, Ny).eval();
  auto const wavelet = psi::wavelets::factory(get_name(DB), LEVEL);
  while(state.KeepRunning())
    wavelet.direct(output, input);
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(Nx) * int64_t(Ny) * sizeof(TYPE));
}

template <class TYPE, unsigned DB = 1, unsigned LEVEL = 1>
void indirect_matrix(benchmark::State &state) {
  auto const Nx = get_size(state.range(0), LEVEL);
  auto const Ny = get_size(state.range(1), LEVEL);
  auto const input = psi::Image<TYPE>::Random(Nx, Ny).eval();
  auto output = psi::Image<TYPE>::Zero(Nx, Ny).eval();
  auto const wavelet = psi::wavelets::factory(get_name(DB), LEVEL);
  while(state.KeepRunning())
    wavelet.indirect(input, output);
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(Nx) * int64_t(Ny) * sizeof(TYPE));
}

template <class TYPE, unsigned DB = 1, unsigned LEVEL = 1>
void direct_vector(benchmark::State &state) {
  auto const Nx = get_size(state.range(0), LEVEL);
  auto const input = psi::Array<TYPE>::Random(Nx).eval();
  auto output = psi::Array<TYPE>::Zero(Nx).eval();
  auto const wavelet = psi::wavelets::factory(get_name(DB), LEVEL);
  while(state.KeepRunning())
    wavelet.direct(output, input);
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(Nx) * sizeof(TYPE));
}
template <class TYPE, unsigned DB = 1, unsigned LEVEL = 1>
void indirect_vector(benchmark::State &state) {
  auto const Nx = get_size(state.range(0), LEVEL);
  auto const input = psi::Array<TYPE>::Random(Nx).eval();
  auto output = psi::Array<TYPE>::Zero(Nx).eval();
  auto const wavelet = psi::wavelets::factory(get_name(DB), LEVEL);
  while(state.KeepRunning())
    wavelet.indirect(input, output);
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(Nx) * sizeof(TYPE));
}

auto const n = 64;
auto const N = 256 * 3;

BENCHMARK_TEMPLATE(direct_matrix, psi::t_complex, 1, 1)->RangePair(n, N, n, N)->UseRealTime();
BENCHMARK_TEMPLATE(direct_matrix, psi::t_real, 1, 1)->RangePair(n, N, n, N)->UseRealTime();
BENCHMARK_TEMPLATE(direct_matrix, psi::t_complex, 10, 1)->RangePair(n, N, n, N)->UseRealTime();

BENCHMARK_TEMPLATE(direct_vector, psi::t_complex, 1, 1)->Range(n, N)->UseRealTime();
BENCHMARK_TEMPLATE(direct_vector, psi::t_complex, 10, 1)->Range(n, N)->UseRealTime();
BENCHMARK_TEMPLATE(direct_vector, psi::t_complex, 1, 2)->Range(n, N)->UseRealTime();
BENCHMARK_TEMPLATE(direct_vector, psi::t_real, 1, 1)->Range(n, N)->UseRealTime();

BENCHMARK_TEMPLATE(indirect_matrix, psi::t_complex, 1, 1)->RangePair(n, N, n, N)->UseRealTime();
BENCHMARK_TEMPLATE(indirect_matrix, psi::t_real, 1, 1)->RangePair(n, N, n, N)->UseRealTime();
BENCHMARK_TEMPLATE(indirect_matrix, psi::t_complex, 10, 1)->RangePair(n, N, n, N)->UseRealTime();

BENCHMARK_TEMPLATE(indirect_vector, psi::t_complex, 1, 1)->Range(n, N)->UseRealTime();
BENCHMARK_TEMPLATE(indirect_vector, psi::t_complex, 10, 1)->Range(n, N)->UseRealTime();
BENCHMARK_TEMPLATE(indirect_vector, psi::t_complex, 1, 2)->Range(n, N)->UseRealTime();
BENCHMARK_TEMPLATE(indirect_vector, psi::t_real, 1, 1)->Range(n, N)->UseRealTime();

BENCHMARK_MAIN();
