// Rank-ordered collectives through one pinned host-memory region that every rank's GPU maps (no peer-to-peer access
// needed, as on RTX 3090s without NVLink): each rank writes its block, raises a flag and reads the others' blocks once
// their flags are up. An element's sum is p_0 + p_1 + ... + p_{W-1} in fp32, in rank order, whichever rank adds it and
// whatever the call's size, so a row's bits never depend on how many rows a call carries; they equal the NCCL
// all-gather followed by the ordered sum (``glue.hc_writeback`` mode 3). Every launch is capturable in a CUDA graph:
// the call's sequence number lives on the device and the last block of a call advances it.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

namespace {

constexpr int MAXW = 8;            // ranks
constexpr int MAXB = 128;          // blocks a call may use (each block raises its own flags)
constexpr int THREADS = 512;
constexpr int PACK = 8;            // elements a slice boundary aligns to (16 bytes of bf16, 32 of fp32)

struct Ctx {
    unsigned char* stage;          // [2 parity][MAXW][cap]: each rank's input
    unsigned char* red;            // [2 parity][MAXW][rcap]: each rank's reduced slice (two-shot)
    uint32_t* flags;               // [2 parity][2 phase][MAXB][MAXW]
    uint32_t* epoch;               // device memory: [0] the last finished call, [1] blocks done in the running one
    long long cap, rcap;           // bytes of one rank's stage and reduced slice
    long long timeout_ns;
    int rank, world;
};

__device__ __forceinline__ uint32_t ld_acquire(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_release(uint32_t* p, uint32_t v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ unsigned long long now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
    return t;
}

// 16 bytes another GPU wrote: never a cached copy of an earlier call's line
__device__ __forceinline__ uint4 ld_shared16(const void* p) {
    return __ldcv(reinterpret_cast<const uint4*>(p));
}

__device__ __forceinline__ uint32_t* flag_row(const Ctx& c, int par, int phase, int b) {
    return c.flags + ((static_cast<size_t>(par) * 2 + phase) * MAXB + b) * MAXW;
}

// Every thread's stores to the region visible system-wide, then this block's flag for (parity, phase) up.
__device__ __forceinline__ void raise_flag(const Ctx& c, int par, int phase, uint32_t ep) {
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) st_release(flag_row(c, par, phase, blockIdx.x) + c.rank, ep);
}

// Until every rank's flag for this block reached ``ep`` (a rank that never arrives traps after the timeout).
__device__ __forceinline__ void wait_flags(const Ctx& c, int par, int phase, uint32_t ep) {
    const int t = threadIdx.x;
    if (t < c.world) {
        const uint32_t* f = flag_row(c, par, phase, blockIdx.x) + t;
        if (static_cast<int32_t>(ld_acquire(f) - ep) < 0) {
            const unsigned long long t0 = now_ns();
            while (static_cast<int32_t>(ld_acquire(f) - ep) < 0) {
                if (now_ns() - t0 > static_cast<unsigned long long>(c.timeout_ns)) {
                    printf("[tensorfold] fastcomm: rank %d waited %.0f s for rank %d (call %u, block %d); trapping\n",
                           c.rank, c.timeout_ns * 1e-9, t, ep, static_cast<int>(blockIdx.x));
                    __trap();
                }
            }
        }
    }
    __syncthreads();
}

// The call's sequence number (all blocks read it before any block of this call can advance it).
__device__ __forceinline__ uint32_t begin(const Ctx& c) {
    return *reinterpret_cast<volatile uint32_t*>(c.epoch) + 1u;
}

// The last block to finish records the call as done, so the next launch on the stream takes the next number.
__device__ __forceinline__ void finish(const Ctx& c, uint32_t ep) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(c.epoch + 1, 1u) == gridDim.x - 1) {
            c.epoch[1] = 0u;
            __threadfence();
            atomicExch(c.epoch, ep);
        }
    }
}

template <typename T> struct Vec;
template <> struct Vec<float> { static constexpr int N = 4; };
template <> struct Vec<__nv_bfloat16> { static constexpr int N = 8; };

// 16 bytes of T -> fp32 values (adds ``n`` of them to acc in order when ADD)
template <typename T>
__device__ __forceinline__ void widen(const uint4& v, float* f);

template <>
__device__ __forceinline__ void widen<float>(const uint4& v, float* f) {
    f[0] = __uint_as_float(v.x); f[1] = __uint_as_float(v.y); f[2] = __uint_as_float(v.z); f[3] = __uint_as_float(v.w);
}

template <>
__device__ __forceinline__ void widen<__nv_bfloat16>(const uint4& v, float* f) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i] = __uint_as_float(w[i] << 16);
        f[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
    }
}

// PACK fp32 sums -> TOut bytes at p (32 bytes of fp32 or 16 of bf16, round to nearest even)
template <typename TOut>
__device__ __forceinline__ void store_pack(void* p, const float* acc);

template <>
__device__ __forceinline__ void store_pack<float>(void* p, const float* acc) {
    float4* q = reinterpret_cast<float4*>(p);
    q[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    q[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
}

template <>
__device__ __forceinline__ void store_pack<__nv_bfloat16>(void* p, const float* acc) {
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __nv_bfloat162 h = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
        w[i] = *reinterpret_cast<const uint32_t*>(&h);
    }
    *reinterpret_cast<uint4*>(p) = make_uint4(w[0], w[1], w[2], w[3]);
}

// PACK elements at element offset e of every rank's stage, summed in rank order (fp32)
template <typename TIn>
__device__ __forceinline__ void sum_pack(const Ctx& c, int par, long long e, float* acc) {
    constexpr int V = Vec<TIn>::N;
    const unsigned char* base = c.stage + static_cast<size_t>(par) * MAXW * c.cap + e * sizeof(TIn);
#pragma unroll
    for (int i = 0; i < PACK; ++i) acc[i] = 0.0f;
    for (int r = 0; r < c.world; ++r) {
        const unsigned char* p = base + static_cast<size_t>(r) * c.cap;
#pragma unroll
        for (int h = 0; h < PACK / V; ++h) {
            float f[V];
            widen<TIn>(ld_shared16(p + h * 16), f);
#pragma unroll
            for (int i = 0; i < V; ++i) {
                if (r == 0) acc[h * V + i] = f[i];               // p_0 as it is, then + p_1 + ... in order
                else acc[h * V + i] = __fadd_rn(acc[h * V + i], f[i]);
            }
        }
    }
}

// Copy [a, b) elements (multiples of PACK) of ``src`` to ``dst``, 16 bytes a thread a step.
template <typename T>
__device__ __forceinline__ void copy_range(unsigned char* dst, const unsigned char* src, long long a, long long b,
                                           bool shared_src) {
    const long long lo = a * static_cast<long long>(sizeof(T)) / 16, hi = b * static_cast<long long>(sizeof(T)) / 16;
    for (long long i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const uint4 v = shared_src ? ld_shared16(src + i * 16) : *reinterpret_cast<const uint4*>(src + i * 16);
        *reinterpret_cast<uint4*>(dst + i * 16) = v;
    }
}

struct Split {
    long long n, sl, cl;           // elements; a rank's slice; a block's chunk of a slice (all multiples of PACK)
};

__device__ __forceinline__ long long clampll(long long v, long long hi) { return v < hi ? v : hi; }

// Two-shot: every rank's input to the region (phase 0), rank r sums slice r over all ranks in order and publishes it
// (phase 1), every rank copies the published slices. Block b handles chunk b of every slice in both phases.
template <typename TIn, typename TOut>
__global__ void __launch_bounds__(THREADS) reduce2_kernel(const TIn* __restrict__ in, TOut* __restrict__ out,
                                                         Ctx c, Split s) {
    const uint32_t ep = begin(c);
    const int par = ep & 1, b = blockIdx.x;
    unsigned char* mine = c.stage + (static_cast<size_t>(par) * MAXW + c.rank) * c.cap;
    for (int sl = 0; sl < c.world; ++sl) {
        const long long a = clampll(sl * s.sl + b * s.cl, s.n), e = clampll(clampll(sl * s.sl + (b + 1) * s.cl,
                                                                                    (sl + 1) * s.sl), s.n);
        if (a < e) copy_range<TIn>(mine, reinterpret_cast<const unsigned char*>(in), a, e, false);
    }
    raise_flag(c, par, 0, ep);
    wait_flags(c, par, 0, ep);
    {
        const long long a = clampll(c.rank * s.sl + b * s.cl, s.n);
        const long long e = clampll(clampll(c.rank * s.sl + (b + 1) * s.cl, (c.rank + 1) * s.sl), s.n);
        unsigned char* red = c.red + (static_cast<size_t>(par) * MAXW + c.rank) * c.rcap;
        for (long long p = a + static_cast<long long>(threadIdx.x) * PACK; p < e; p += static_cast<long long>(blockDim.x) * PACK) {
            float acc[PACK];
            sum_pack<TIn>(c, par, p, acc);
            store_pack<TOut>(red + (p - c.rank * s.sl) * sizeof(TOut), acc);
            store_pack<TOut>(out + p, acc);
        }
    }
    raise_flag(c, par, 1, ep);
    wait_flags(c, par, 1, ep);
    for (int r = 0; r < c.world; ++r) {
        if (r == c.rank) continue;
        const long long a = clampll(r * s.sl + b * s.cl, s.n);
        const long long e = clampll(clampll(r * s.sl + (b + 1) * s.cl, (r + 1) * s.sl), s.n);
        if (a >= e) continue;
        const unsigned char* red = c.red + (static_cast<size_t>(par) * MAXW + r) * c.rcap;
        // the published slice starts at element r * sl: shift so element p sits at (p - r * sl)
        copy_range<TOut>(reinterpret_cast<unsigned char*>(out) + static_cast<size_t>(r * s.sl) * sizeof(TOut),
                         red, a - r * s.sl, e - r * s.sl, true);
    }
    finish(c, ep);
}

// One-shot: every rank's input to the region, then every rank sums block b's chunk over all ranks in order.
template <typename TIn, typename TOut>
__global__ void __launch_bounds__(THREADS) reduce1_kernel(const TIn* __restrict__ in, TOut* __restrict__ out,
                                                         Ctx c, Split s) {
    const uint32_t ep = begin(c);
    const int par = ep & 1, b = blockIdx.x;
    unsigned char* mine = c.stage + (static_cast<size_t>(par) * MAXW + c.rank) * c.cap;
    const long long a = clampll(b * s.cl, s.n), e = clampll((b + 1) * s.cl, s.n);
    if (a < e) copy_range<TIn>(mine, reinterpret_cast<const unsigned char*>(in), a, e, false);
    raise_flag(c, par, 0, ep);
    wait_flags(c, par, 0, ep);
    for (long long p = a + static_cast<long long>(threadIdx.x) * PACK; p < e; p += static_cast<long long>(blockDim.x) * PACK) {
        float acc[PACK];
        sum_pack<TIn>(c, par, p, acc);
        store_pack<TOut>(out + p, acc);
    }
    finish(c, ep);
}

// All-gather of ``words`` 4-byte words a rank: out[r * words + i] = rank r's in[i], block b moving chunk b.
__global__ void __launch_bounds__(THREADS) gather_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ out,
                                                         Ctx c, long long words, long long chunk) {
    const uint32_t ep = begin(c);
    const int par = ep & 1, b = blockIdx.x;
    uint32_t* mine = reinterpret_cast<uint32_t*>(c.stage + (static_cast<size_t>(par) * MAXW + c.rank) * c.cap);
    const long long a = clampll(b * chunk, words), e = clampll((b + 1) * chunk, words);
    for (long long i = a + threadIdx.x; i < e; i += blockDim.x) mine[i] = in[i];
    raise_flag(c, par, 0, ep);
    wait_flags(c, par, 0, ep);
    for (int r = 0; r < c.world; ++r) {
        const uint32_t* theirs = reinterpret_cast<const uint32_t*>(c.stage + (static_cast<size_t>(par) * MAXW + r) * c.cap);
        for (long long i = a + threadIdx.x; i < e; i += blockDim.x)
            out[r * words + i] = __ldcv(reinterpret_cast<const unsigned int*>(theirs + i));
    }
    finish(c, ep);
}

Ctx context(int64_t region, int64_t epoch, int64_t cap, int64_t rcap, int64_t rank, int64_t world, double timeout_s) {
    Ctx c;
    unsigned char* base = reinterpret_cast<unsigned char*>(region);
    c.flags = reinterpret_cast<uint32_t*>(base);
    const size_t flag_bytes = static_cast<size_t>(2) * 2 * MAXB * MAXW * sizeof(uint32_t);
    c.stage = base + flag_bytes;
    c.red = c.stage + static_cast<size_t>(2) * MAXW * cap;
    c.epoch = reinterpret_cast<uint32_t*>(epoch);
    c.cap = cap;
    c.rcap = rcap;
    c.rank = static_cast<int>(rank);
    c.world = static_cast<int>(world);
    c.timeout_ns = static_cast<long long>(timeout_s * 1e9);
    return c;
}

int blocks_for(long long bytes, int most) {
    const long long per = 16 * 1024;                 // bytes a block moves per rank before another block helps
    long long b = (bytes + per - 1) / per;
    b = b < 1 ? 1 : b;
    return static_cast<int>(b < most ? b : most);
}

} // namespace

int64_t fastcomm_region_bytes(int64_t cap, int64_t rcap) {
    return static_cast<int64_t>(2) * 2 * MAXB * MAXW * sizeof(uint32_t) + 2 * MAXW * (cap + rcap);
}

// out = the rank-ordered fp32 sum of every rank's ``in`` (fp32 or bf16), stored as out's dtype (fp32 or bf16).
void fastcomm_reduce_cuda(const at::Tensor& in, at::Tensor& out, int64_t region, const at::Tensor& epoch, int64_t cap,
                          int64_t rcap, int64_t rank, int64_t world, int64_t mode, int64_t blocks, double timeout_s) {
    const long long n = in.numel();
    TORCH_CHECK(in.is_cuda() && out.is_cuda() && in.is_contiguous() && out.is_contiguous() && out.numel() == n,
                "fastcomm: contiguous CUDA in and out of the same size");
    TORCH_CHECK((in.scalar_type() == at::kFloat || in.scalar_type() == at::kBFloat16) &&
                (out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16), "fastcomm: fp32 or bf16");
    TORCH_CHECK(n % PACK == 0 && reinterpret_cast<uintptr_t>(in.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "fastcomm: a multiple of 8 elements, 16-byte aligned");
    Ctx c = context(region, reinterpret_cast<int64_t>(epoch.data_ptr()), cap, rcap, rank, world, timeout_s);
    auto stream = at::cuda::getCurrentCUDAStream();
    const bool two = mode == 2;
    Split s;
    s.n = n;
    s.sl = two ? ((n + world - 1) / world + PACK - 1) / PACK * PACK : n;
    const int most = blocks > 0 ? static_cast<int>(blocks) : MAXB;
    const int B = blocks_for((two ? s.sl : n) * static_cast<long long>(in.element_size()), most < MAXB ? most : MAXB);
    s.cl = ((s.sl + B - 1) / B + PACK - 1) / PACK * PACK;
    TORCH_CHECK(n * in.element_size() <= cap, "fastcomm: ", n * in.element_size(), " bytes past the region's ", cap);
    TORCH_CHECK(!two || s.sl * out.element_size() <= rcap, "fastcomm: a reduced slice past the region's ", rcap);
#define GO(K, TI, TO) K<TI, TO><<<B, THREADS, 0, stream>>>(reinterpret_cast<const TI*>(in.data_ptr()), \
                                                           reinterpret_cast<TO*>(out.data_ptr()), c, s)
    const bool fin = in.scalar_type() == at::kFloat, fout = out.scalar_type() == at::kFloat;
    if (two) {
        if (fin && fout) GO(reduce2_kernel, float, float);
        else if (fin) GO(reduce2_kernel, float, __nv_bfloat16);
        else if (fout) GO(reduce2_kernel, __nv_bfloat16, float);
        else GO(reduce2_kernel, __nv_bfloat16, __nv_bfloat16);
    } else {
        if (fin && fout) GO(reduce1_kernel, float, float);
        else if (fin) GO(reduce1_kernel, float, __nv_bfloat16);
        else if (fout) GO(reduce1_kernel, __nv_bfloat16, float);
        else GO(reduce1_kernel, __nv_bfloat16, __nv_bfloat16);
    }
#undef GO
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fastcomm_gather_cuda(const at::Tensor& in, at::Tensor& out, int64_t region, const at::Tensor& epoch, int64_t cap,
                          int64_t rcap, int64_t rank, int64_t world, int64_t blocks, double timeout_s) {
    const long long bytes = in.numel() * in.element_size();
    const long long words = bytes / 4;
    TORCH_CHECK(in.is_cuda() && out.is_cuda() && in.is_contiguous() && out.is_contiguous() && bytes % 4 == 0 &&
                out.numel() * out.element_size() == world * bytes, "fastcomm gather: out holds world x in");
    TORCH_CHECK(bytes <= cap, "fastcomm: ", bytes, " bytes past the region's ", cap);
    Ctx c = context(region, reinterpret_cast<int64_t>(epoch.data_ptr()), cap, rcap, rank, world, timeout_s);
    const int most = blocks > 0 ? static_cast<int>(blocks) : MAXB;
    const int B = blocks_for(bytes, most < MAXB ? most : MAXB);
    const long long chunk = (words + B - 1) / B;
    gather_kernel<<<B, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint32_t*>(in.data_ptr()), reinterpret_cast<uint32_t*>(out.data_ptr()), c, words, chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
