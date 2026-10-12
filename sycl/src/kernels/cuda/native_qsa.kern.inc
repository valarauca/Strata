// native_qsa.kern.inc - entry points of native_qsa.dp.cpp (strata/ze/kernel_abi.hpp).
#define NQ_NORM(B)                                                                                                  \
    STRATA_KERNEL(native_qsa_norm_##B, (const float* input, const float* gamma, float* output, int n_cols,          \
                                        float epsilon),                                                            \
                  norm<B>(input, gamma, output, n_cols, epsilon))                                                  \
    STRATA_KERNEL(native_qsa_norm_grouped_##B, (const float* input, const float* gamma, float* output, int n_cols, \
                                                int groups, float epsilon),                                        \
                  norm_grouped<B>(input, gamma, output, n_cols, groups, epsilon))
NQ_NORM(256)
NQ_NORM(1024)
#undef NQ_NORM
STRATA_KERNEL(native_qsa_gate, (const float* attn, const float* q_full, float* output, int n_head, int head_dim),
              gate(attn, q_full, output, n_head, head_dim))
