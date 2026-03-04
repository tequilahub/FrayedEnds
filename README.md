# FrayedEnds 

FrayedEnds is a framework that discretizes continuous quantum systems and transforms their continuous Hamiltonians into an optimal second quantized form through adaptive orbital optimization. Fast and accurate calculations are performed using a Multi-Resolution-Analysis representation (implemented with [MADNESS](https://github.com/m-a-d-n-e-s-s/madness)).



### Attention: This code is a first version that may still contain various bugs and has not yet been cleaned up/optimized.


- The included devcontainer automatically installs all necessary system packages, madness, conda and all necessary python packages to compile the code and run it in combination with Tequila(VQE) or Block2(DMRG).
- Information about development containers and their installation in combination with VSCode can be found at: [https://code.visualstudio.com/docs/devcontainers/containers](https://code.visualstudio.com/docs/devcontainers/containers).

# Installation
FrayedEnds supports macOS and linux.
The setup process depends on the operating system. 

## Linux



## Step 1: install madness
```bash
MADNESS_DIR=realpath madness
git clone https://github.com/m-a-d-n-e-s-s/madness.git madness_source
mkdir madness_build
cmake -D CMAKE_INSTALL_PREFIX=$MADNESS_DIR -DENABLE_MPI=OFF -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" -S madness_source -B madness_build
make -C madness_build -j8
cmake --build madness_build/ --target install -j8
```

## Step 2: install the interface 
```bash
MADNESS_DIR=$MADNESS_DIR pip install -e .
```

