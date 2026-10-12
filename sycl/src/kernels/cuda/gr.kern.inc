// gr.kern.inc - entry points of gr.dp.cpp (strata/ze/kernel_abi.hpp).
STRATA_KERNEL(gr_norm_f32, (const float* R, const float* w_norm, float eps, int n_embd, float* xn, uint16_t* xq),
              gr_norm_kernel<true>(R, w_norm, eps, n_embd, xn, xq))
STRATA_KERNEL(gr_norm_bf16, (const float* R, const float* w_norm, float eps, int n_embd, float* xn, uint16_t* xq),
              gr_norm_kernel<false>(R, w_norm, eps, n_embd, xn, xq))
STRATA_KERNEL(gr_down_f32, (const float* xq, const uint16_t* w_down, int hc_dim, int hc_lr, int hc, float* lq),
              gr_down_kernel<float>(xq, w_down, hc_dim, hc_lr, hc, lq))
STRATA_KERNEL(gr_down_bf16, (const uint16_t* xq, const uint16_t* w_down, int hc_dim, int hc_lr, int hc, uint16_t* lq),
              gr_down_kernel<uint16_t>(xq, w_down, hc_dim, hc_lr, hc, lq))
STRATA_KERNEL(gr_gate_f32, (const float* lq, const uint16_t* w_up, const float* xn, int hc_dim, int hc_lr, float* gated),
              gr_gate_kernel<float>(lq, w_up, xn, hc_dim, hc_lr, gated))
STRATA_KERNEL(gr_gate_bf16,
              (const uint16_t* lq, const uint16_t* w_up, const float* xn, int hc_dim, int hc_lr, float* gated),
              gr_gate_kernel<uint16_t>(lq, w_up, xn, hc_dim, hc_lr, gated))
STRATA_KERNEL(gr_mean, (const float* gated, int n_embd, int hc, float* mixed), gr_mean_kernel(gated, n_embd, hc, mixed))
STRATA_KERNEL(gr_inject_f32, (const float* xq, const uint16_t* w_inject, int hc_dim, int hc, float* inject),
              gr_inject_kernel<float>(xq, w_inject, hc_dim, hc, inject))
STRATA_KERNEL(gr_inject_bf16, (const uint16_t* xq, const uint16_t* w_inject, int hc_dim, int hc, float* inject),
              gr_inject_kernel<uint16_t>(xq, w_inject, hc_dim, hc, inject))
STRATA_KERNEL(gr_write, (const float* R, const float* block_out, const float* inject, int n_embd, int hc, float* out),
              gr_write_kernel(R, block_out, inject, n_embd, hc, out, strata_slm<uint8_t>()))
