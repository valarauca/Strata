# Strata's Intel engine - build and run image for the Intel Arc path.
# Base: the llama.cpp SYCL image already on the box (oneAPI 2026.1 compiler for the offline device modules, the Level
# Zero loader and the Xe2 compute runtime the engine runs on). The engine links no SYCL runtime; icpx is the build's
# device compiler only.
FROM ghcr.io/snailium/llama.cpp-sycl-intel-b70/llama-sycl-b70:stable
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      ninja-build unzip python3-pip python3-venv rsync \
    && rm -rf /var/lib/apt/lists/*
ENV ONEAPI_DEVICE_SELECTOR=level_zero:0 ZES_ENABLE_SYSMAN=1
ENTRYPOINT ["/bin/bash", "-lc"]
