//==---- util.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_UTIL_HPP__
#define __DPCT_UTIL_HPP__

#include <sycl/sycl.hpp>
#include <complex>
#include <type_traits>
#include <cassert>
#include <cstdint>

// TODO: Remove these function definitions once they exist in the DPC++ compiler
#if defined(__SYCL_DEVICE_ONLY__) && defined(__INTEL_LLVM_COMPILER)
template <typename T>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL __SYCL_EXPORT __attribute__((noduplicate))
T __spirv_GroupNonUniformShuffle(__spv::Scope::Flag, T, unsigned) noexcept;

template <typename T>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL __SYCL_EXPORT __attribute__((noduplicate))
T __spirv_GroupNonUniformShuffleDown(__spv::Scope::Flag, T, unsigned) noexcept;

template <typename T>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL __SYCL_EXPORT __attribute__((noduplicate))
T __spirv_GroupNonUniformShuffleUp(__spv::Scope::Flag, T, unsigned) noexcept;
#endif

namespace dpct {

/// Reverse the bit order of an unsigned integer
/// \param [in] a Input unsigned integer value
/// \returns Value of a with the bit order reversed
template <typename T> inline T reverse_bits(T a) {
  static_assert(std::is_unsigned<T>::value && std::is_integral<T>::value,
                "unsigned integer required");
  if (!a)
    return 0;
  T mask = 0;
  size_t count = 4 * sizeof(T);
  mask = ~mask >> count;
  while (count) {
    a = ((a & mask) << count) | ((a & ~mask) >> count);
    count = count >> 1;
    mask = mask ^ (mask << count);
  }
  return a;
}

/// \param [in] a The first value contains 4 bytes
/// \param [in] b The second value contains 4 bytes
/// \param [in] s The selector value, only lower 16bit used
/// \returns the permutation result of 4 bytes selected in the way
/// specified by \p s from \p a and \p b
inline unsigned int byte_level_permute(unsigned int a, unsigned int b,
                                       unsigned int s) {
  unsigned int ret;
  ret =
      ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff) << 8) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff) << 16) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff) << 24);
  return ret;
}

/// \param [in] low32 The 4 bytes to construct the 8 bytes value as low 32 bits.
/// \param [in] high32 The 4 bytes to construct the 8 bytes value as high 32
/// bits.
/// \param [in] sel The selector value. It is used to generate selectors which are
/// used to fetch byte value from \p low32 and \p high32 to construct result
/// value.
/// \param [in] mode The mode of permutation, together with \p sel, it
/// further defines the behavior of data selection. \p mode and \p sel define 4
/// selectors (s[i], i=0, 1, 2, 3), which are used as index to fetch 4 bytes from
/// the 8 bytes value constructed by \p low32 and \p high32; the byte selected by s[i]
/// is used to fill the i-th byte of the result.
/// The available mode values are:
/// mode value 0: s[i] = sel[i * 4 + 3 : i * 4]
/// mode value 1: s[i] = sel[1 : 0] + i
/// mode value 2: s[i] = (sel[1 : 0] + 7) % 8
/// mode value 3: s[i] = sel[1 : 0]
/// mode value 4: s[i] = max(sel[1 : 0], i)
/// mode value 5: s[i] = min(sel[1 : 0], i)
/// mode value 6: s[0] = sel[0 : 0] * 2
///               s[1] = sel[0 : 0] * 2 + 1
///               s[2] = s[0]
///               s[3] = s[1]
/// other value: illegal, undefined behavior, return 0.
inline uint32_t byte_level_permute_custom(uint32_t low32, uint32_t high32,
                                          uint32_t sel, int mode = 0) {
  constexpr uint16_t lookup[6][4] = {
      {0x3210, 0x4321, 0x5432, 0x6543}, // Forward 4-byte extract
      {0x5670, 0x6701, 0x7012, 0x0123}, // Backward 4-byte extract
      {0x0000, 0x1111, 0x2222, 0x3333}, // Replicate 8-bit values
      {0x3210, 0x3211, 0x3222, 0x3333}, // Edge clamp left
      {0x0000, 0x1110, 0x2210, 0x3210}, // Edge clamp right
      {0x1010, 0x3232, 0x1010, 0x3232}  // Replicate 16-bit values
  };

  if (mode >= 1 && mode <= 6) {
    return byte_level_permute(low32, high32, lookup[mode - 1][sel & 0x3]);
  } else if (!mode) {
    return byte_level_permute(low32, high32, sel);
  }
  return 0;
}

/// Find position of first least significant set bit in an integer.
/// ffs(0) returns 0.
///
/// \param [in] a Input integer value
/// \returns The position
template <typename T> inline int ffs(T a) {
  static_assert(std::is_integral<T>::value, "integer required");
  return (sycl::ctz(a) + 1) % (sizeof(T) * 8 + 1);
}

/// select_from_sub_group allows work-items to obtain a copy of a value held by
/// any other work-item in the sub_group. The input sub_group will be divided
/// into several logical sub_groups with id range [0, \p logical_sub_group_size
/// - 1]. Each work-item in logical sub_group gets value from another work-item
/// whose id is \p remote_local_id. If \p remote_local_id is outside the
/// logical sub_group id range, \p remote_local_id will modulo with \p
/// logical_sub_group_size. The \p logical_sub_group_size must be a power of 2
/// and not exceed input sub_group size.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] remote_local_id Input source work item id
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T select_from_sub_group(sycl::sub_group g, T x, int remote_local_id,
                        int logical_sub_group_size = 32) {
  unsigned int start_index =
      g.get_local_linear_id() / logical_sub_group_size * logical_sub_group_size;
  return sycl::select_from_group(
      g, x, start_index + remote_local_id % logical_sub_group_size);
}

/// shift_sub_group_left move values held by the work-items in a sub_group
/// directly to another work-item in the sub_group, by shifting values a fixed
/// number of work-items to the left. The input sub_group will be divided into
/// several logical sub_groups with id range [0, \p logical_sub_group_size - 1].
/// Each work-item in logical sub_group gets value from another work-item whose
/// id is caller's id adds \p delta. If calculated id is outside the logical
/// sub_group id range, the work-item will get value from itself. The \p
/// logical_sub_group_size must be a power of 2 and not exceed input sub_group
/// size.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] delta Input delta
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T shift_sub_group_left(sycl::sub_group g, T x, unsigned int delta,
                       int logical_sub_group_size = 32) {
  unsigned int id = g.get_local_linear_id();
  unsigned int end_index =
      (id / logical_sub_group_size + 1) * logical_sub_group_size;
  T result = sycl::shift_group_left(g, x, delta);
  if ((id + delta) >= end_index) {
    result = x;
  }
  return result;
}

/// shift_sub_group_right move values held by the work-items in a sub_group
/// directly to another work-item in the sub_group, by shifting values a fixed
/// number of work-items to the right. The input sub_group will be divided into
/// several logical_sub_groups with id range [0, \p logical_sub_group_size - 1].
/// Each work-item in logical_sub_group gets value from another work-item whose
/// id is caller's id subtracts \p delta. If calculated id is outside the
/// logical sub_group id range, the work-item will get value from itself. The \p
/// logical_sub_group_size must be a power of 2 and not exceed input sub_group
/// size.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] delta Input delta
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T shift_sub_group_right(sycl::sub_group g, T x, unsigned int delta,
                        int logical_sub_group_size = 32) {
  unsigned int id = g.get_local_linear_id();
  unsigned int start_index =
      id / logical_sub_group_size * logical_sub_group_size;
  T result = sycl::shift_group_right(g, x, delta);
  if ((id - start_index) < delta) {
    result = x;
  }
  return result;
}

/// permute_sub_group_by_xor permutes values by exchanging values held by pairs
/// of work-items identified by computing the bitwise exclusive OR of the
/// work-item id and some fixed mask. The input sub_group will be divided into
/// several logical sub_groups with id range [0, \p logical_sub_group_size - 1].
/// Each work-item in logical sub_group gets value from another work-item whose
/// id is bitwise exclusive OR of the caller's id and \p mask. If calculated id
/// is outside the logical sub_group id range, the work-item will get value from
/// itself. The \p logical_sub_group_size must be a power of 2 and not exceed
/// input sub_group size.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] mask Input mask
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T permute_sub_group_by_xor(sycl::sub_group g, T x, unsigned int mask,
                           int logical_sub_group_size = 32) {
  unsigned int id = g.get_local_linear_id();
  unsigned int start_index =
      id / logical_sub_group_size * logical_sub_group_size;
  unsigned int target_offset = (id % logical_sub_group_size) ^ mask;
  return sycl::select_from_group(g, x,
                                 target_offset < logical_sub_group_size
                                     ? start_index + target_offset
                                     : id);
}

/// The function match_any_over_sub_group conducts a comparison of values
/// across work-items within a sub-group. match_any_over_sub_group return a mask
/// in which some bits are set to 1, indicating that the \p value provided by
/// the work-item represented by these bits are equal. The n-th bit of mask
/// representing the work-item with id n. The parameter \p member_mask
/// indicating the work-items participating the call.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] member_mask Input mask
/// \param [in] value Input value
/// \returns The result
template <typename T>
unsigned int match_any_over_sub_group(sycl::sub_group g, unsigned member_mask,
                                      T value) {
  static_assert(std::is_trivially_copyable_v<T>,
                "Value type must be trivially copyable type.");
  if (!member_mask) {
    return 0;
  }
  unsigned int id = g.get_local_linear_id();
  unsigned int flag = 0, result = 0, reduce_result = 0;
  unsigned int bit_index = 0x1 << id;
  bool is_participate = member_mask & bit_index;
  T broadcast_value = 0;
  bool matched = false;
  while (flag != member_mask) {
    broadcast_value =
        sycl::select_from_group(g, value, sycl::ctz((~flag & member_mask)));
    reduce_result = sycl::reduce_over_group(
        g, is_participate ? (broadcast_value == value ? bit_index : 0) : 0,
        sycl::plus<>());
    flag |= reduce_result;
    matched = reduce_result & bit_index;
    result = matched * reduce_result + (1 - matched) * result;
  }
  return result;
}

/// The function match_all_over_sub_group conducts a comparison of values
/// across work-items within a sub-group. match_all_over_sub_group return \p
/// member_mask and predicate \p pred will be set to 1 if all \p value that
/// provided by each work-item in \p member_mask are equal, otherwise return 0
/// and the predicate \p pred will be set to 0. The n-th bit of \p member_mask
/// representing the work-item with id n. The parameter \p member_mask
/// indicating the work-items participating the call.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] member_mask Input mask
/// \param [in] value Input value
/// \param [out] pred Output predicate
/// \returns The result
template <typename T>
unsigned int match_all_over_sub_group(sycl::sub_group g, unsigned member_mask,
                                      T value, int *pred) {
  static_assert(std::is_trivially_copyable_v<T>,
                "Value type must be trivially copyable type.");
  if (!member_mask) {
    return 0;
  }
  unsigned int id = g.get_local_linear_id();
  unsigned int bit_index = 0x1 << id;
  bool is_participate = member_mask & bit_index;
  T broadcast_value = sycl::select_from_group(g, value, sycl::ctz(member_mask));
  unsigned int reduce_result = sycl::reduce_over_group(
      g,
      (member_mask & bit_index) ? (broadcast_value == value ? bit_index : 0)
                                : 0,
      sycl::plus<>());
  bool all_equal = (reduce_result == member_mask);
  *pred = is_participate & all_equal;
  return all_equal * member_mask;
}

namespace experimental {
#if defined(__NVPTX__) && defined(__SYCL_DEVICE_ONLY__)
#define SHFL_SYNC(RES, MASK, VAL, SHFL_PARAM, C, SHUFFLE_INSTR)                \
  if constexpr (std::is_same_v<T, double>) {                                   \
    int x_a, x_b;                                                              \
    asm("mov.b64 {%0,%1},%2;" : "=r"(x_a), "=r"(x_b) : "d"(VAL));              \
    auto tmp_a = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, x_a, SHFL_PARAM, C);   \
    auto tmp_b = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, x_b, SHFL_PARAM, C);   \
    asm("mov.b64 %0,{%1,%2};" : "=d"(RES) : "r"(tmp_a), "r"(tmp_b));           \
  } else if constexpr (std::is_same_v<T, long> ||                              \
                       std::is_same_v<T, unsigned long>) {                     \
    int x_a, x_b;                                                              \
    asm("mov.b64 {%0,%1},%2;" : "=r"(x_a), "=r"(x_b) : "l"(VAL));              \
    auto tmp_a = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, x_a, SHFL_PARAM, C);   \
    auto tmp_b = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, x_b, SHFL_PARAM, C);   \
    asm("mov.b64 %0,{%1,%2};" : "=l"(RES) : "r"(tmp_a), "r"(tmp_b));           \
  } else if constexpr (std::is_same_v<T, sycl::half>) {                        \
    short tmp_b16;                                                             \
    asm("mov.b16 %0,%1;" : "=h"(tmp_b16) : "h"(VAL));                          \
    auto tmp_b32 = __nvvm_shfl_sync_##SHUFFLE_INSTR(                           \
        MASK, static_cast<int>(tmp_b16), SHFL_PARAM, C);                       \
    asm("mov.b16 %0,%1;" : "=h"(RES) : "h"(static_cast<short>(tmp_b32)));      \
  } else if constexpr (std::is_same_v<T, float>) {                             \
    auto tmp_b32 = __nvvm_shfl_sync_##SHUFFLE_INSTR(                           \
        MASK, __nvvm_bitcast_f2i(VAL), SHFL_PARAM, C);                         \
    RES = __nvvm_bitcast_i2f(tmp_b32);                                         \
  } else {                                                                     \
    RES = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, VAL, SHFL_PARAM, C);          \
  }
#endif
/// Masked version of select_from_sub_group, which execute masked sub-group
/// operation. The parameter member_mask indicating the work-items participating
/// the call. Whether the n-th bit is set to 1 representing whether the
/// work-item with id n is participating the call. All work-items named in
/// member_mask must be executed with the same member_mask, or the result is
/// undefined.
/// \tparam T Input value type
/// \param [in] member_mask Input mask
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] remote_local_id Input source work item id
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T select_from_sub_group(unsigned int member_mask,
                        sycl::sub_group g, T x, int remote_local_id,
                        int logical_sub_group_size = 32) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__SPIR__)
  unsigned int start_index =
      g.get_local_linear_id() / logical_sub_group_size * logical_sub_group_size;
  unsigned logical_remote_id =
      start_index + remote_local_id % logical_sub_group_size;
  return __spirv_GroupNonUniformShuffle(__spv::Scope::Subgroup, x, logical_remote_id);
#elif defined(__NVPTX__)
  T result;
  int cVal = ((32 - logical_sub_group_size) << 8) | 31;
  SHFL_SYNC(result, member_mask, x, remote_local_id, cVal, idx_i32)
  return result;
#endif
#else
  (void)g;
  (void)x;
  (void)remote_local_id;
  (void)logical_sub_group_size;
  (void)member_mask;
  throw sycl::exception(sycl::errc::runtime, "Masked version of select_from_sub_group not "
                        "supported on host device.");
#endif // __SYCL_DEVICE_ONLY__
}

/// Masked version of shift_sub_group_left, which execute masked sub-group
/// operation. The parameter member_mask indicating the work-items participating
/// the call. Whether the n-th bit is set to 1 representing whether the
/// work-item with id n is participating the call. All work-items named in
/// member_mask must be executed with the same member_mask, or the result is
/// undefined.
/// \tparam T Input value type
/// \param [in] member_mask Input mask
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] delta Input delta
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T shift_sub_group_left(unsigned int member_mask,
                       sycl::sub_group g, T x, unsigned int delta,
                       int logical_sub_group_size = 32) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__SPIR__)
  unsigned int id = g.get_local_linear_id();
  unsigned int end_index =
      (id / logical_sub_group_size + 1) * logical_sub_group_size;
  T result = __spirv_GroupNonUniformShuffleDown(__spv::Scope::Subgroup, x, delta);
  if ((id + delta) >= end_index) {
    result = x;
  }
  return result;
#elif defined(__NVPTX__)
  T result;
  int cVal = ((32 - logical_sub_group_size) << 8) | 31;
  SHFL_SYNC(result, member_mask, x, delta, cVal, down_i32)
  return result;
#endif
#else
  (void)g;
  (void)x;
  (void)delta;
  (void)logical_sub_group_size;
  (void)member_mask;
  throw sycl::exception(sycl::errc::runtime, "Masked version of shift_sub_group_left not "
                        "supported on host device.");
#endif // __SYCL_DEVICE_ONLY__
}

/// Masked version of shift_sub_group_right, which execute masked sub-group
/// operation. The parameter member_mask indicating the work-items participating
/// the call. Whether the n-th bit is set to 1 representing whether the
/// work-item with id n is participating the call. All work-items named in
/// member_mask must be executed with the same member_mask, or the result is
/// undefined.
/// \tparam T Input value type
/// \param [in] member_mask Input mask
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] delta Input delta
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T shift_sub_group_right(unsigned int member_mask,
                        sycl::sub_group g, T x, unsigned int delta,
                        int logical_sub_group_size = 32) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__SPIR__)
  unsigned int id = g.get_local_linear_id();
  unsigned int start_index =
      id / logical_sub_group_size * logical_sub_group_size;
  T result = __spirv_GroupNonUniformShuffleUp(__spv::Scope::Subgroup, x, delta);
  if ((id - start_index) < delta) {
    result = x;
  }
  return result;
#elif defined(__NVPTX__)
  T result;
  int cVal = ((32 - logical_sub_group_size) << 8);
  SHFL_SYNC(result, member_mask, x, delta, cVal, up_i32)
  return result;
#endif
#else
  (void)g;
  (void)x;
  (void)delta;
  (void)logical_sub_group_size;
  (void)member_mask;
  throw sycl::exception(sycl::errc::runtime, "Masked version of shift_sub_group_right not "
                        "supported on host device.");
#endif // __SYCL_DEVICE_ONLY__
}

/// Masked version of shift_sub_group_left, which execute masked sub-group
/// operation. The parameter member_mask indicating the work-items participating
/// the call. Whether the n-th bit is set to 1 representing whether the
/// work-item with id n is participating the call. All work-items named in
/// member_mask must be executed with the same member_mask, or the result is
/// undefined.
/// \tparam LogicSubGroupSize Input logical sub_group size
/// \tparam T Input value type
/// \param [in] member_mask Input mask
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] delta Input delta
/// \param [in] last_item Index of last thread in logical subgroup
/// \returns The result
template <size_t LogicSubGroupSize, typename T>
T shift_sub_group_left(sycl::sub_group sg, T input, int delta, int last_item,
                       unsigned member_mask) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  T result;
  constexpr int cVal = ((32 - LogicSubGroupSize) << 8);
  SHFL_SYNC(result, member_mask, input, delta, cVal | last_item, down_i32)
  return result;
#else
  if ((1U << sg.get_local_linear_id()) & member_mask) {
    auto inner = sycl::ext::oneapi::experimental::entangle(sg);
    sycl::group_barrier(inner);
  }
  auto partition =
      sycl::ext::oneapi::experimental::chunked_partition<LogicSubGroupSize>(
          sg);
  int id = partition.get_local_linear_id();
  T result = sycl::shift_group_left(partition, input, delta);
  if ((id + delta) > last_item)
    result = input;
  return result;
#endif
}

/// Masked version of shift_sub_group_right, which execute masked sub-group
/// operation. The parameter member_mask indicating the work-items participating
/// the call. Whether the n-th bit is set to 1 representing whether the
/// work-item with id n is participating the call. All work-items named in
/// member_mask must be executed with the same member_mask, or the result is
/// undefined.
/// \tparam LogicSubGroupSize Input logical sub_group size
/// \tparam T Input value type
/// \param [in] member_mask Input mask
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] delta Input delta
/// \param [in] first_item Index of first lane in logical subgroup
/// \returns The result
template <size_t LogicSubGroupSize, typename T>
T shift_sub_group_right(sycl::sub_group sg, T input, int delta, int first_item,
                        unsigned member_mask) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  T result;
  constexpr int cVal = ((32 - LogicSubGroupSize) << 8);
  SHFL_SYNC(result, member_mask, input, delta, cVal | first_item, up_i32)
  return result;
#else
  if ((1U << sg.get_local_linear_id()) & member_mask) {
    auto inner = sycl::ext::oneapi::experimental::entangle(sg);
    sycl::group_barrier(inner);
  }
  auto partition =
      sycl::ext::oneapi::experimental::chunked_partition<LogicSubGroupSize>(
          sg);
  int id = sg.get_local_linear_id();
  T result = sycl::shift_group_right(partition, input, delta);
  if ((id - first_item) < delta)
    result = input;
  return result;
#endif
}

/// Masked version of permute_sub_group_by_xor, which execute masked sub-group
/// operation. The parameter member_mask indicating the work-items participating
/// the call. Whether the n-th bit is set to 1 representing whether the
/// work-item with id n is participating the call. All work-items named in
/// member_mask must be executed with the same member_mask, or the result is
/// undefined.
/// \tparam T Input value type
/// \param [in] member_mask Input mask
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] mask Input mask
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T permute_sub_group_by_xor(unsigned int member_mask,
                           sycl::sub_group g, T x, unsigned int mask,
                           int logical_sub_group_size = 32) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__SPIR__)
  unsigned int id = g.get_local_linear_id();
  unsigned int start_index =
      id / logical_sub_group_size * logical_sub_group_size;
  unsigned int target_offset = (id % logical_sub_group_size) ^ mask;
  unsigned logical_remote_id = (target_offset < logical_sub_group_size) ? start_index + target_offset : id;
  return __spirv_GroupNonUniformShuffle(__spv::Scope::Subgroup, x, logical_remote_id);
#elif defined(__NVPTX__)
  T result;
  int cVal = ((32 - logical_sub_group_size) << 8) | 31;
  SHFL_SYNC(result, member_mask, x, mask, cVal, bfly_i32)
  return result;
#endif
#else
  (void)g;
  (void)x;
  (void)mask;
  (void)logical_sub_group_size;
  (void)member_mask;
  throw sycl::exception(sycl::errc::runtime, "Masked version of permute_sub_group_by_xor not "
                        "supported on host device.");
#endif // __SYCL_DEVICE_ONLY__
}
#if defined(__NVPTX__)
#undef SHFL_SYNC
#endif
} // namespace experimental

} // namespace dpct

#endif // __DPCT_UTIL_HPP__
