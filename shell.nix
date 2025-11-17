{
  pkgs ? import <nixpkgs> { config.allowUnfree = true; },
}:

let
  cuda = pkgs.cudaPackages_12_4.cudatoolkit;
  cudaLibs = pkgs.cudaPackages_12_4.cuda_cudart;
in
pkgs.mkShell {
  name = "cuda-env-shell";

  buildInputs = with pkgs; [
    gcc11
    gnumake
    cmake
    cuda
    cudaLibs
    linuxPackages.nvidia_x11
    cudaPackages_12_4.nsight_systems
    cudaPackages_12_4.nsight_compute
  ];

  shellHook = ''
    export CUDA_PATH=${cuda}
    export PATH=${cuda}/bin:$PATH
    export LD_LIBRARY_PATH=${cuda}/lib64:${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH
    export LIBRARY_PATH=${cuda}/lib64:$LIBRARY_PATH
    export CPLUS_INCLUDE_PATH=${cuda}/include:$CPLUS_INCLUDE_PATH
    export CUDAHOSTCXX=${pkgs.gcc11}/bin/g++
    export NVCC_PREPEND_FLAGS="-ccbin ${pkgs.gcc11}/bin/g++"
    echo "CUDA toolkit: ${cuda}"
  '';
}
