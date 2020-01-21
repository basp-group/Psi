
# Psi

## Description

Psi is an open-source package that implements convex optimisation algorithms. It is designed to provide support for primal-dual simulations in the Puri-Psi package.

## Installation

### C++ pre-requisites and dependencies

- [CMake](http://www.cmake.org/): Program building software.
- [tiff](http://www.libtiff.org/): Tag Image File Format library
- [OpenMP](http://openmp.org/wp/): Optional. Parallelises operations across multiple threads.
- [MPI](http://www.mpi-forum.org): Optional. Parallelises operations across multiple processes.
- [Eigen 3](http://eigen.tuxfamily.org/index.php?title=Main_Page): Modern C++ linear algebra.
- [spdlog](https://github.com/gabime/spdlog): Logging library.
- [Catch2](https://github.com/catchorg/Catch2): A C++ unit-testing framework.
- [google/benchmark](https://github.com/google/benchmark): Micro-benchmarking framework.

### Installing

Once the dependencies are present, the program can be built with:

```
cd /path/to/code
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

 ## Contributors

Psi has been developed by:

- [Adrian Jackson](https://www.epcc.ed.ac.uk/about/staff/mr-adrian-jackson)
- Pierre-Antoine Thouvenin
- Ming Jiang
- Alex Onose
- [Yves Wiaux](http://basp.eps.hw.ac.uk/)

Psi started life as a fork of the [Sopt](https://github.com/basp-group/sopt) software package, developed in collaboration with UCL.

## References and citation


## Acknowledgements

## License


```
Psi: Copyright (C) 2015-2020

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details (LICENSE.txt).

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
```
