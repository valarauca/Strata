// device.kern.inc - entry points of device.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(device_poison, (float* p, uint64_t n_floats), poison_kernel(p, n_floats))
