#include <psi/maths.h>
#include <psi/types.h>

int main(int, char const **) {
  psi::Image<std::complex<int>> input(2, 2), weights(2, 2);
  input << 1, -2, 3, -4;
  weights << 5, 6, 7, 8;

  if(psi::l1_norm(input, weights) != 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8)
    throw std::exception();

  return 0;
}
