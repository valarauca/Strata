// src/kernels/cuda/native_qsa_score.host.cpp - the host side of native_qsa_score.dp.cpp: checks and the launcher.
//
// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{mmf.cuh,mma.cuh,unary.cu,binbcast.cu}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/native_qsa_score.hpp"
#include "strata/ze/kernel_abi.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {

#include "native_qsa_score.kern.inc"

namespace {
std::atomic<bool> enabled{false};
constexpr int D=128, HEADS=4, R=4, ROWS=32, WARPS=2;
struct Span{const void* p;size_t n;};
void validate(Span s){
    const auto p=reinterpret_cast<uintptr_t>(s.p);
    if(!p||p%4||s.n>UINTPTR_MAX-p)throw std::invalid_argument("native QSA score requires aligned bounded spans");
}
bool overlaps(Span a,Span b){
    const auto x=reinterpret_cast<uintptr_t>(a.p),y=reinterpret_cast<uintptr_t>(b.p);
    return x<y+b.n&&y<x+a.n;
}
} // namespace
void native_qsa_score_set_enabled(bool value){enabled.store(value,std::memory_order_relaxed);}
bool native_qsa_score_enabled(){return enabled.load(std::memory_order_relaxed);}
void native_qsa_score(const float* pooled,const float* query,const float* bias,
                      const QsaShapes& s,const int32_t* step,int64_t max_blocks,int64_t max_cells,
                      float* cells,void* stream){
    if(!stream||s.idx_dim!=D||s.idx_n_head!=HEADS||s.idx_block!=R||s.idx_top_k!=2048||
       max_cells<1||max_cells>INT32_MAX-3||max_blocks!=max_cells/R+1)
        throw std::invalid_argument("native QSA score requires128dim/4heads/4cells/2048budget, exact capacities and explicit stream");
    const Span spans[]={{pooled,size_t(max_blocks)*D*4},{query,HEADS*D*4},
        {step,kStepCount*4},{cells,size_t(max_cells)*4},{bias,bias?size_t(max_blocks)*4:0}};
    const int count=bias?5:4;
    for(int i=0;i<count;++i)validate(spans[i]);
    for(int i=0;i<count;++i)for(int j=i+1;j<count;++j)
        if(overlaps(spans[i],spans[j]))throw std::invalid_argument("native QSA score spans overlap");
    k_native_qsa_score(ze::stream_of(stream), {unsigned((max_blocks + ROWS - 1) / ROWS), 1, 1}, {32, WARPS, 1}, 0,
                       pooled, query, bias, step, int(max_cells), cells);
}
} // namespace strata::kernels
