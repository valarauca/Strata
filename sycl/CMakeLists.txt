# Strata on Intel GPUs: the Intel engine, Strata's own engine on Level Zero.
#
# This is a separate project on purpose. The top-level build describes the CUDA/HIP engine; every source it lists is
# taken from ../src unless this directory carries the Intel copy. A kernel family `src/kernels/cuda/X.cu` is three
# files here: `X.dp.cpp` (device code, compiled offline into native GPU modules), `X.kern.inc` (its entry points) and
# `X.host.cpp` (the launchers). Host files that called the CUDA runtime have an Intel copy (`src/X.cpp`) that calls
# the Level Zero runtime in src/ze.
#
#   cmake -S sycl -B build-ze -G Ninja -DCMAKE_CXX_COMPILER=icpx && cmake --build build-ze
#
# There is no SYCL runtime in the program: host code is plain C++ (icpx without -fsycl), the device code is compiled
# by the SYCL device compiler alone (cmake/ze_device.cmake) and launched through Level Zero (src/ze).
cmake_minimum_required(VERSION 3.24)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt _strata_proj REGEX "^project\\(strata VERSION [0-9.]+")
string(REGEX MATCH "VERSION ([0-9.]+)" _m "${_strata_proj}")
if(CMAKE_MATCH_1)
  set(STRATA_SYCL_VERSION ${CMAKE_MATCH_1})
else()
  set(STRATA_SYCL_VERSION 0.1.40.2)
endif()
project(strata_intel VERSION ${STRATA_SYCL_VERSION} LANGUAGES C CXX ASM)
set(STRATA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}" STRATA_INTEL_ZE=1)

# The GPU the device modules are compiled for (ocloc -device). STRATA_SYCL_AOT is the old spelling, kept so existing
# build commands keep working.
set(STRATA_SYCL_AOT "bmg-g31" CACHE STRING "old name of STRATA_ZE_DEVICE")
set(STRATA_ZE_DEVICE "${STRATA_SYCL_AOT}" CACHE STRING "ocloc device of the GPU modules (bmg-g31: Arc Pro B70)")
if(NOT STRATA_ZE_DEVICE)
  message(FATAL_ERROR "STRATA_ZE_DEVICE is empty: the device modules are compiled ahead of time for one GPU (bmg-g31)")
endif()
set(STRATA_SYCL_SPIN_MAX "" CACHE STRING "fixed bound of the device's spins on host flags, in reads (empty: chosen per device at run time)")
set(_strata_defs "")
if(STRATA_SYCL_SPIN_MAX)
  list(APPEND _strata_defs STRATA_SYCL_SPIN_MAX=${STRATA_SYCL_SPIN_MAX}u)
endif()
if(STRATA_ZE_DEVICE MATCHES "^(bmg|lnl)")
  list(APPEND _strata_defs STRATA_XMX_SG16=1)   # the grouped XMX kernels on sub-group 16, joint_matrix 8x16x16 (Xe2)
endif()
add_compile_definitions(${_strata_defs})
# icpx defaults to a fast floating-point model; the host code (rope tables, CPU references, sampling on the host) was
# measured with the precise one, as was the device code.
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fp-model=precise>")
endif()
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>" "$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-variable>"
                    "$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ze_device.cmake)
strata_ze_device_includes(${CMAKE_CURRENT_SOURCE_DIR}/include ${STRATA_ROOT}/include ${STRATA_ROOT}/third_party/ggml
                          ${STRATA_ROOT}/src/kernels ${STRATA_ROOT}/src/core)
strata_ze_device_defines(STRATA_VERSION="${PROJECT_VERSION}" STRATA_INTEL_ZE=1 ${_strata_defs})

# --- source resolution: the Intel copy when there is one, the original otherwise ---------------------------------
function(strata_resolve out)
  set(r "")
  foreach(f IN LISTS ARGN)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${f})
      list(APPEND r ${CMAKE_CURRENT_SOURCE_DIR}/${f})
    elseif(EXISTS ${STRATA_ROOT}/${f})
      list(APPEND r ${STRATA_ROOT}/${f})
    else()
      message(FATAL_ERROR "no such source: ${f}")
    endif()
  endforeach()
  set(${out} ${r} PARENT_SCOPE)
endfunction()

find_library(STRATA_ZE_LOADER ze_loader REQUIRED)

add_library(strata_includes INTERFACE)
target_include_directories(strata_includes INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include      # the Intel copies of the engine's headers, first
  ${STRATA_ROOT}/include)
target_compile_options(strata_includes INTERFACE -Wall -Wextra)

# --- CPU-side pieces, identical to the top-level build ------------------------------------------------------------
strata_resolve(_spec src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp
                    src/spec/draft_source.cpp)
add_library(strata_spec STATIC ${_spec})
target_link_libraries(strata_spec PUBLIC strata_includes)

strata_resolve(_cpu src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp src/kernels/cpu/expert_layout.cpp
                    src/kernels/cpu/q2_avx2.cpp)
add_library(strata_kernels_cpu STATIC ${_cpu})
set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS
  "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
# AVX-VNNI copies of the AVX-2 expert kernels (q2_avx2.cpp, iq_avx2.cpp), taken at run time where the CPU has them
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma -mf16c")
check_cxx_source_compiles([=[
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avxvnni")))
#endif
static __m256i dots(__m256i a, __m256i b) { return _mm256_dpbusd_avx_epi32(_mm256_dpwssd_avx_epi32(a, a, b), a, b); }
int main() {
    volatile int x = 1;
    return _mm256_extract_epi32(dots(_mm256_set1_epi32(x), _mm256_set1_epi32(x)), 0) == 0;
}
]=] STRATA_HAVE_AVXVNNI_INTRINSICS)
unset(CMAKE_REQUIRED_FLAGS)
if(STRATA_HAVE_AVXVNNI_INTRINSICS)
  set(_strata_avxvnni STRATA_AVXVNNI=1)
else()
  set(_strata_avxvnni STRATA_AVXVNNI=0)
endif()
set_property(SOURCE ${STRATA_ROOT}/src/kernels/cpu/q2_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_strata_avxvnni})
find_package(Threads REQUIRED)
target_link_libraries(strata_kernels_cpu PUBLIC strata_includes Threads::Threads)

option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
if(STRATA_NATIVE_EXPERTS)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_SYCL OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  if(STRATA_GGML_DIR)
    add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(strata_llamacpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
      GIT_SHALLOW FALSE)
    FetchContent_GetProperties(strata_llamacpp)
    if(NOT strata_llamacpp_POPULATED)
      FetchContent_Populate(strata_llamacpp)
    endif()
    add_subdirectory(${strata_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  endif()
  strata_resolve(_cpu_native src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/iq_avx2.cpp
                             src/kernels/cpu/kq_avx2.cpp src/kernels/cpu/kq_avx1.cpp   # 0.1.31: the Q4_K/Q5_K/Q5_1/Q8_0 expert rows
                             src/kernels/cpu/q2_avx1.cpp src/kernels/cpu/iq_avx1.cpp)   # 0.1.42: AVX1 (no AVX2) Q2_0 / i-quant rows
  target_sources(strata_kernels_cpu PRIVATE ${_cpu_native})
  target_include_directories(strata_kernels_cpu PRIVATE ${STRATA_ROOT}/third_party/ggml)
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS
    "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "-mavx")
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/q2_avx1.cpp ${STRATA_ROOT}/src/kernels/cpu/iq_avx1.cpp PROPERTIES
    COMPILE_OPTIONS "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
  set_property(SOURCE ${STRATA_ROOT}/src/kernels/cpu/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_strata_avxvnni})
  target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
endif()

# --- the device code: one bundle of native modules per kernel family -----------------------------------------------
# <dir>/<X>.dp.cpp + <X>.kern.inc (device) and <X>.host.cpp (launchers).
set(STRATA_KERNEL_FAMILIES dequant_s2 s2_gemv s_gemv s2_gemv_quads s2_gemv_fast router_top10 quantize_act rope sampler
    s2_gemv_q8 s2_expert_grouped shared_expert gr gdn qsa elementwise bf16_gemv native_mmvq native_bf16 native_gr_norm
    native_moe native_gdn native_gr_postops native_router native_gdn_preprocess native_qsa native_rope native_ple_postops
    native_flash_attn native_qsa_indexer native_qsa_score ple kv_q8 kv_stream kv_q4 fused_gr cvec qsa_decode_attn
    qsa_prompt_attn fused_gdn dequant_bf16 qsa_select verify_kernels iq_kernels qsa_prompt_attn_xmx)
set(STRATA_PREFILL_FAMILIES gemm kernels xmx_moe)
set(STRATA_CORE_FAMILIES device)
# families whose kernels need the large register file (256 GRF): the XMX GEMMs (grf_size<256> on a free-function
# kernel drops it, so the whole bundle gets the ocloc option)
set(STRATA_OCLOC_gemm "-ze-opt-large-register-file")
set(STRATA_OCLOC_xmx_moe "-ze-opt-large-register-file")
set(_kernel_host "")
set(_prefill_host "")
set(_core_host "")
function(strata_family dir fam hostvar)
  set(base ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/${fam})
  foreach(f ${base}.dp.cpp ${base}.kern.inc ${base}.host.cpp)
    if(NOT EXISTS ${f})
      message(FATAL_ERROR "kernel family ${dir}/${fam}: ${f} is missing")
    endif()
  endforeach()
  strata_ze_bundle(${fam} SOURCES ${base}.dp.cpp KERN ${base}.kern.inc OCLOC_OPTIONS "${STRATA_OCLOC_${fam}}")
  set(${hostvar} ${${hostvar}} ${base}.host.cpp PARENT_SCOPE)
endfunction()
foreach(k IN LISTS STRATA_KERNEL_FAMILIES)
  strata_family(src/kernels/cuda ${k} _kernel_host)
endforeach()
foreach(k IN LISTS STRATA_PREFILL_FAMILIES)
  strata_family(src/prefill ${k} _prefill_host)
endforeach()
foreach(k IN LISTS STRATA_CORE_FAMILIES)
  strata_family(src/core ${k} _core_host)
endforeach()
strata_ze_bundle(iq_xmx SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/iq_xmx.dp.cpp
                 KERN ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/iq_xmx.kern.inc OCLOC_OPTIONS "-ze-opt-large-register-file")
strata_ze_bundle(ze_core SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/ze/ze_core.dp.cpp
                 KERN ${CMAKE_CURRENT_SOURCE_DIR}/src/ze/ze_core.kern.inc)

# --- the Level Zero runtime: devices, the context, memory, streams, events, graphs, modules -----------------------
add_library(strata_ze STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/ze/ze.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/ze/modules.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/ze/ze_util.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/ze/intel_gpu.cpp)
target_link_libraries(strata_ze PUBLIC strata_includes ${STRATA_ZE_LOADER} Threads::Threads)
strata_ze_embed(strata_ze)

# --- the GPU engine (host side) ----------------------------------------------------------------------------------
strata_resolve(_core src/core/pinned.cpp src/platform/memory.cpp src/core/graph.cpp src/core/weights.cpp
                     src/core/layout.cpp)
add_library(strata_core STATIC ${_core} ${_core_host})
target_link_libraries(strata_core PUBLIC strata_includes strata_ze)

strata_resolve(_kernels src/kernels/ngram.cpp src/ngram/ple_reader.cpp src/platform/direct_file.cpp)
add_library(strata_kernels STATIC ${_kernel_host} ${_kernels})
target_include_directories(strata_kernels PRIVATE ${STRATA_ROOT}/third_party/ggml)
target_link_libraries(strata_kernels PUBLIC strata_core)

strata_resolve(_engine src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp src/core/remote_experts.cpp
                       src/core/expert_cache.cpp src/core/native_head.cpp src/core/native_dense.cpp src/core/verify.cpp
                       src/core/mtp.cpp src/core/conversation_snapshot.cpp src/core/conversation_state.cpp
                       src/core/conversation_memory.cpp
                       src/core/arch_defaults.cpp src/core/vmm.cpp
                       src/core/conversation_file.cpp
                       src/core/peer_experts.cpp src/core/remote_expert_opt_stub.cpp)
add_library(strata_engine STATIC ${_engine} ${CMAKE_CURRENT_SOURCE_DIR}/src/core/gguf_expert_source.cpp)
target_include_directories(strata_engine PRIVATE ${STRATA_ROOT}/src/core)   # conversation_checked.hpp (#include "...")
target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu)

strata_resolve(_prefill src/prefill/prefill.cpp)
add_library(strata_prefill STATIC ${_prefill} ${_prefill_host}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/moe_fused_stub.cpp)   # upstream's fused int8 kernels: not built here
target_link_libraries(strata_prefill PUBLIC strata_engine)

strata_resolve(_main src/program/generate.cpp)
add_executable(strata ${_main})
target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec)

# the prompt path's XMX GEMM against a double-precision CPU product at the model's shapes (--bench: its speed)
add_executable(xgemm_parity ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/xgemm_parity.cpp)
target_link_libraries(xgemm_parity PRIVATE strata_prefill)

# --- the kernel parity self-tests (each runs on the card and compares against a CPU reference) ---------------------
option(STRATA_SYCL_PARITY "Build the kernel parity tests" ON)
if(STRATA_SYCL_PARITY)
  enable_testing()
  foreach(p IN ITEMS dequant_s2 s2_gemv shared_expert gr gdn s2_gemv_q8 sampler rope quantize_act router_top10
                     s_gemv elementwise bf16_gemv s_gemv_q8k qsa kv_q8 kv_stream kv_q4 cvec)
    strata_resolve(_src src/kernels/${p}_parity.cpp)
    add_executable(${p}_parity ${_src})
    target_include_directories(${p}_parity PRIVATE ${STRATA_ROOT}/src/kernels)   # the oracle .inc fixtures
    target_link_libraries(${p}_parity PRIVATE strata_kernels)
    add_test(NAME ${p}_parity COMMAND ${p}_parity --selftest)
  endforeach()
  # #1795: iq_parity and ple_parity are registered as the top-level CMakeLists.txt does, not by the loop above
  # (iq_parity takes a fixture directory, not --selftest; ple_parity reads real artifacts from the engine root).
  foreach(p IN ITEMS iq ple)
    strata_resolve(_src src/kernels/${p}_parity.cpp)
    add_executable(${p}_parity ${_src})
    target_include_directories(${p}_parity PRIVATE ${STRATA_ROOT}/src/kernels)
    target_link_libraries(${p}_parity PRIVATE strata_kernels)
  endforeach()
  # iq: tools/iq_fixture.py writes the fixtures into the build tree (needs python3 with numpy; exit 3 = skip)
  set(_iq_python "")
  set(_iq_candidates "")
  if(IQ_FIXTURE_PYTHON)
    list(APPEND _iq_candidates ${IQ_FIXTURE_PYTHON})
  else()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
      list(APPEND _iq_candidates ${Python3_EXECUTABLE})
    endif()
    list(APPEND _iq_candidates "${STRATA_ROOT}/.venv/bin/python" "${STRATA_ROOT}/.venv/Scripts/python.exe" /usr/bin/python3 python3)
  endif()
  foreach(_cand IN LISTS _iq_candidates)
    if(_cand STREQUAL "")
      continue()
    endif()
    execute_process(COMMAND ${_cand} -c "import numpy" OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE _iq_probe)
    if(_iq_probe EQUAL 0)
      set(_iq_python "${_cand}")
      break()
    endif()
  endforeach()
  if(NOT _iq_python STREQUAL "")
    add_test(NAME iq_parity_fixtures COMMAND ${_iq_python} ${STRATA_ROOT}/tools/iq_fixture.py
                                             --out ${CMAKE_CURRENT_BINARY_DIR}/iq_fixture)
    set_tests_properties(iq_parity_fixtures PROPERTIES FIXTURES_SETUP iq_fixture_data SKIP_RETURN_CODE 3)
    add_test(NAME iq_parity COMMAND iq_parity ${CMAKE_CURRENT_BINARY_DIR}/iq_fixture)
    set_tests_properties(iq_parity PROPERTIES FIXTURES_REQUIRED iq_fixture_data SKIP_RETURN_CODE 3)
  else()
    message(STATUS "STRATA SYCL tests: no python3 with numpy found - iq_parity and its fixtures are not registered")
  endif()
  # ple: real inputs (the Q2_0 shard-2 GGUF, the dense pack, the ggml capture); a missing one is a hard error INSIDE
  # the test, as in the top-level build.  The working directory is the engine root, so the relative defaults resolve
  # there.  Point it at your files with -DSTRATA_PLE_GGUF=<shard 2 .gguf> and -DSTRATA_PLE_FIXTURE_DIR=<dir>.
  set(STRATA_PLE_GGUF "" CACHE FILEPATH "Q2_0 shard 2 GGUF that ple_parity reads (default: ../../Q2_0/... from the engine root)")
  set(STRATA_PLE_FIXTURE_DIR "${STRATA_ROOT}/bench/micro" CACHE PATH
      "Directory containing required ple_in.bin and ple_out.bin parity fixtures")
  set(_ple_args --selftest --in "${STRATA_PLE_FIXTURE_DIR}/ple_in.bin" --out "${STRATA_PLE_FIXTURE_DIR}/ple_out.bin")
  if(STRATA_PLE_GGUF)
    list(APPEND _ple_args --gguf "${STRATA_PLE_GGUF}")
  endif()
  add_test(NAME ple_parity COMMAND ple_parity ${_ple_args} WORKING_DIRECTORY ${STRATA_ROOT})
  # 0.1.31: the IQ kernels that decode a weight part once for every column/entry, bitwise vs the per-column ones
  # (no --selftest flag: it runs its checks by default), and the rewritten per-hit/grouped expert kernels vs the
  # previous ones, bitwise
  strata_resolve(_src src/kernels/iq_multi_parity.cpp)
  add_executable(iq_multi_parity ${_src})
  target_link_libraries(iq_multi_parity PRIVATE strata_kernels)
  add_test(NAME iq_multi_parity COMMAND iq_multi_parity)
  strata_resolve(_src src/kernels/s2_expert_grouped_parity.cpp)
  add_executable(s2_expert_grouped_parity ${_src})
  target_link_libraries(s2_expert_grouped_parity PRIVATE strata_kernels)
  add_test(NAME s2_expert_grouped_parity COMMAND s2_expert_grouped_parity --selftest)
  # upstream PR #363: native_expert_grouped's group stride and fused SwiGLU + q8_1, bitwise vs the v1 launches
  strata_resolve(_src src/kernels/native_grouped_parity.cpp)
  add_executable(native_grouped_parity ${_src})
  target_link_libraries(native_grouped_parity PRIVATE strata_kernels)
  add_test(NAME native_grouped_parity COMMAND native_grouped_parity)
  # 0.1.40 (#783): the batched verify-window kernels against the ones they replace, bitwise (synthetic, no model)
  foreach(p IN ITEMS verify native_multi kv_steps resident_plan gr_multi)
    strata_resolve(_src src/kernels/${p}_parity.cpp)
    add_executable(${p}_parity ${_src})
    target_include_directories(${p}_parity PRIVATE ${STRATA_ROOT}/src/kernels)
    target_link_libraries(${p}_parity PRIVATE strata_kernels)
    add_test(NAME ${p}_parity COMMAND ${p}_parity)
  endforeach()
  # not built here: qsa_topk_parity / qsa_topk_active_parity (the sm_90 thread-block-cluster QSA top-k has no Intel
  # counterpart: the callers take the plain kernels); the SYCL-runtime benches, probes and parity programs that were not
  # moved to Level Zero (native_expert_parity, q5_projection_parity, route_window_parity, the mmvq/dequant/gr/xmx
  # benches) were removed with the SYCL runtime; xgemm_parity covers the prompt path's GEMM
  # (upstream PR #413's gdn_rec_parity is an SM-holding NVIDIA bench - PTX timers, __nanosleep - not built here; the
  # key-head kernel is checked through the engine: output hashes with STRATA_GDN_KEYHEAD=1 and without)
  # the conversation snapshots (#216) against the engine, as upstream's STRATA_BUILD_CONVERSATION_TESTS
  strata_resolve(_src src/core/conversation_snapshot_test.cpp)
  add_executable(conversation_snapshot_test ${_src})
  target_link_libraries(conversation_snapshot_test PRIVATE strata_engine)
  add_test(NAME conversation_snapshot_test COMMAND conversation_snapshot_test)
  set_tests_properties(conversation_snapshot_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
  # the prompt-attention parity takes a context length, not --selftest, and needs the tensor-core kernel (not
  # ported yet: it reports "refused the pools" here). Built, not registered.
  strata_resolve(_src src/kernels/qsa_prompt_attn_parity.cpp)
  add_executable(qsa_prompt_attn_parity ${_src})
  target_link_libraries(qsa_prompt_attn_parity PRIVATE strata_kernels)
  # the hybrid K8V4 KV test (upstream 0.1.27) passes its appends, gathers and attention here and fails only its
  # last step, the same tensor-core prompt kernel. Built, not registered, until that kernel is ported.
  strata_resolve(_src src/kernels/kv_hybrid_parity.cpp)
  add_executable(kv_hybrid_parity ${_src})
  target_include_directories(kv_hybrid_parity PRIVATE ${STRATA_ROOT}/src/kernels)
  target_link_libraries(kv_hybrid_parity PRIVATE strata_kernels)
  # the multi_exact contract of native_mmvq, bitwise (upstream 0.1.40). Built, not registered.
  strata_resolve(_src src/kernels/mmvq_multi_parity.cpp)
  add_executable(mmvq_multi_parity ${_src})
  target_link_libraries(mmvq_multi_parity PRIVATE strata_kernels)
endif()
