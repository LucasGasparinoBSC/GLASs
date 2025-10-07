# GPU Library for Accelerated Solvers (GLASs)

## About

GLASs is a high-performance library designed to provide easy-to-use GPU-accelerated iterative algebraic solvers for large scale computing problems, including linear systems and eigenvalue problems.

Its main feature is to allow client codes to interact with the solvers without adapting to library data structures: a solver call simply requires the passing of a client-side subroutine that computes the matrix-vector product, or other operations as needed by the solver (boundary conditions, preconditioners, etc.). The user routine can be either CPU or GPU capable, including usage of OpenACC directives for GPU scenarios.

GLASs is implemented in C++ and CUDA C, and provides wrappers for Fortran client interactions.

## Dependencies

Generic:
- C++17 compatible compiler
- Modern Fortran compiler
- CMake 3.15 or higher
- MPI library (e.g., OpenMPI, MPICH)

GPU support:
- NVIDIA GPU with CUDA capability 6.0 or higher
- CUDA Toolkit 11.0 or higher
- NVIDIA HPC SDK compilers

## Building/Installing

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GLASs.git
   ```

2. Create a build directory and navigate into it:
   ```bash
   mkdir GLASs/build
   cd GLASs/build
   ```

3. Configure the build with CMake:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/install
   ```

For GPU support:

1. Ensure that the CUDA Toolkit and NVIDIA HPC SDK are installed and properly configured in your environment. Recommend using modulefiles if available.
2. Ensure that a NVIDIA GPU is available on and visible to the system (e.g., using `nvidia-smi`).
3. CMake configuration:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/install -DUSE_GPU=ON
   ```

4. Build and install the library:
   ```bash
   make -j$(nproc)
   make install
   ```

Optionally, run tests to verify the installation:
   ```bash
   ctest
   ```

## Usage

Once the library is installed, you can link against it in your C++ or Fortran projects. Ensure that your LIBRARY_PATH and LD_LIBRARY_PATH see the installed GLASs library. Once again, we suggest the usage of modulefiles to manage environment variables.

GLASs is built as a series of dynamic libraries, which can be linked to your application. The main library is `libIterSolvers.so`, which contains the core iterative solvers. Additional libraries provide support for specific functionalities, such as preconditioners or GPU acceleration. These can be used as needed in the client code as well, without any support provided.