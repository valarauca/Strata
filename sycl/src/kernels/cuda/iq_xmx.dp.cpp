// src/kernels/cuda/iq_xmx.dp.cpp - iq_kernels.dp.cpp's grouped XMX kernels as their own device bundle, so that only
// they get the large register file (iq_xmx.kern.inc).
#define STRATA_IQ_XMX_MODULE
#include "iq_kernels.dp.cpp"
