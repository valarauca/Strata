# cmake/ze_device.cmake - the engine's device code, compiled offline into native Level Zero modules.
#
# strata_ze_bundle(<name> SOURCES <dev.cpp...> [KERN <x.kern.inc...>] [OCLOC_OPTIONS "..."]) compiles each source with the SYCL device
# compiler alone (icpx -fsycl-device-only: no host code, no SYCL runtime), then tools/ze_devlink.py links the group,
# splits it where kernels need different properties, compiles it for STRATA_ZE_DEVICE with ocloc and writes
# <name>.szeb. -D/-U options in CMAKE_CXX_FLAGS reach the device compile too, so host and device agree on every
# compile-time knob. OCLOC_OPTIONS adds module-wide compile options (-ze-opt-large-register-file: 256 GRF for every
# kernel of the bundle; a per-kernel SYCL property for it drops a free-function kernel). strata_ze_embed(<target> <bundles...>) links the bundles into <target> as read-only data.
include_guard(GLOBAL)

find_program(STRATA_ICPX icpx REQUIRED)
find_program(STRATA_OCLOC ocloc REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
get_filename_component(_strata_icpx_real "${STRATA_ICPX}" REALPATH)
get_filename_component(_strata_icpx_bin "${_strata_icpx_real}" DIRECTORY)
set(STRATA_ZE_COMPILER_DIR "${_strata_icpx_bin}/compiler" CACHE PATH "icpx's internal tool directory (llvm-link, sycl-post-link, llvm-spirv)")

set(STRATA_ZE_DEVICE_FLAGS -fsycl -fsycl-device-only -fsycl-targets=spir64_gen -fsycl-default-sub-group-size=32
    -fno-sycl-dead-args-optimization -O3 -DNDEBUG -fp-model=precise -std=c++20 -Wno-unused-parameter
    -Wno-unused-variable -Wno-deprecated-declarations -Wno-unknown-pragmas -Wno-ignored-attributes)
set(STRATA_ZE_OCLOC_OPTIONS "-cl-fp32-correctly-rounded-divide-sqrt" CACHE STRING "ocloc -options for the device modules")

set_property(GLOBAL PROPERTY STRATA_ZE_INCLUDES "")
set_property(GLOBAL PROPERTY STRATA_ZE_DEFINES "")
function(strata_ze_device_includes)
  set_property(GLOBAL APPEND PROPERTY STRATA_ZE_INCLUDES ${ARGN})
endfunction()
function(strata_ze_device_defines)
  set_property(GLOBAL APPEND PROPERTY STRATA_ZE_DEFINES ${ARGN})
endfunction()

function(strata_ze_bundle name)
  cmake_parse_arguments(B "" "OCLOC_OPTIONS" "SOURCES;KERN" ${ARGN})
  get_property(incs GLOBAL PROPERTY STRATA_ZE_INCLUDES)
  get_property(defs GLOBAL PROPERTY STRATA_ZE_DEFINES)
  set(iflags "")
  foreach(i IN LISTS incs)
    list(APPEND iflags "-I${i}")
  endforeach()
  set(dflags "")
  foreach(d IN LISTS defs)
    list(APPEND dflags "-D${d}")
  endforeach()
  separate_arguments(cxx_flags NATIVE_COMMAND "${CMAKE_CXX_FLAGS}")
  set(split_flag "")
  foreach(f IN LISTS cxx_flags)
    if(split_flag)
      list(APPEND dflags "${split_flag}${f}")
      set(split_flag "")
    elseif(f STREQUAL "-D" OR f STREQUAL "-U")
      set(split_flag "${f}")
    elseif(f MATCHES "^-[DU]")
      list(APPEND dflags "${f}")
    endif()
  endforeach()
  set(cppargs "")
  foreach(d IN LISTS dflags)
    list(APPEND cppargs "--cpp-arg=${d}")
  endforeach()
  set(dir ${CMAKE_BINARY_DIR}/ze/${name})
  file(MAKE_DIRECTORY ${dir})
  set(bcs "")
  foreach(src IN LISTS B_SOURCES)
    get_filename_component(stem ${src} NAME_WE)
    set(bc ${dir}/${stem}.bc)
    add_custom_command(OUTPUT ${bc}
      COMMAND ${STRATA_ICPX} ${STRATA_ZE_DEVICE_FLAGS} ${dflags} ${iflags} -MD -MF ${bc}.d -MT ${bc} -c ${src} -o ${bc}
      DEPENDS ${src}
      DEPFILE ${bc}.d
      COMMENT "device ${name}: ${stem}"
      VERBATIM)
    list(APPEND bcs ${bc})
  endforeach()
  set(expect "")
  foreach(k IN LISTS B_KERN)
    list(APPEND expect --expect ${k})
  endforeach()
  set(opts "${STRATA_ZE_OCLOC_OPTIONS}")
  if(B_OCLOC_OPTIONS)
    string(APPEND opts " ${B_OCLOC_OPTIONS}")
  endif()
  set(out ${CMAKE_BINARY_DIR}/ze/${name}.szeb)
  add_custom_command(OUTPUT ${out}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/ze_devlink.py --out ${out} --work ${dir}/link
            --device ${STRATA_ZE_DEVICE} --ocloc-options=${opts} --compiler-dir ${STRATA_ZE_COMPILER_DIR}
            --ocloc ${STRATA_OCLOC} --cpp ${STRATA_ICPX} ${cppargs} ${expect} ${bcs}
    DEPENDS ${bcs} ${B_KERN} ${CMAKE_CURRENT_SOURCE_DIR}/tools/ze_devlink.py
    COMMENT "device ${name}: link + ocloc ${STRATA_ZE_DEVICE}"
    VERBATIM)
  set_property(GLOBAL APPEND PROPERTY STRATA_ZE_BUNDLES ${out})
  set_property(GLOBAL APPEND PROPERTY STRATA_ZE_BUNDLE_NAMES ${name})
endfunction()

# Links every bundle declared so far into <target> (an assembly file with one .incbin per bundle and the tables
# src/ze/modules.cpp reads).
function(strata_ze_embed target)
  get_property(bundles GLOBAL PROPERTY STRATA_ZE_BUNDLES)
  get_property(names GLOBAL PROPERTY STRATA_ZE_BUNDLE_NAMES)
  list(LENGTH bundles n)
  set(asm ${CMAKE_BINARY_DIR}/ze/ze_bundles.S)
  set(s "\t.section .rodata.strata_ze,\"a\"\n")
  set(i 0)
  foreach(b IN LISTS bundles)
    string(APPEND s "\t.balign 64\nstrata_ze_b${i}:\n\t.incbin \"${b}\"\nstrata_ze_b${i}_end:\n")
    math(EXPR i "${i} + 1")
  endforeach()
  set(i 0)
  foreach(nm IN LISTS names)
    string(APPEND s "strata_ze_n${i}:\n\t.asciz \"${nm}\"\n")
    math(EXPR i "${i} + 1")
  endforeach()
  string(APPEND s "\t.balign 8\n\t.global strata_ze_nbundles\nstrata_ze_nbundles:\n\t.long ${n}\n")
  string(APPEND s "\t.balign 8\n\t.global strata_ze_bundle_size\nstrata_ze_bundle_size:\n")
  set(i 0)
  foreach(b IN LISTS bundles)
    string(APPEND s "\t.quad strata_ze_b${i}_end - strata_ze_b${i}\n")
    math(EXPR i "${i} + 1")
  endforeach()
  string(APPEND s "\t.section .data.rel.ro.strata_ze,\"aw\"\n\t.balign 8\n\t.global strata_ze_bundle_data\nstrata_ze_bundle_data:\n")
  set(i 0)
  foreach(b IN LISTS bundles)
    string(APPEND s "\t.quad strata_ze_b${i}\n")
    math(EXPR i "${i} + 1")
  endforeach()
  string(APPEND s "\t.global strata_ze_bundle_name\nstrata_ze_bundle_name:\n")
  set(i 0)
  foreach(b IN LISTS bundles)
    string(APPEND s "\t.quad strata_ze_n${i}\n")
    math(EXPR i "${i} + 1")
  endforeach()
  string(APPEND s "\t.section .note.GNU-stack,\"\",@progbits\n")
  file(GENERATE OUTPUT ${asm} CONTENT "${s}")
  set_source_files_properties(${asm} PROPERTIES OBJECT_DEPENDS "${bundles}" LANGUAGE ASM)
  target_sources(${target} PRIVATE ${asm})
endfunction()
