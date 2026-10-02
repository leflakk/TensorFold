#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <string>

int64_t fastcomm_region_bytes(int64_t, int64_t);
void fastcomm_reduce_cuda(const at::Tensor&, at::Tensor&, int64_t, const at::Tensor&, int64_t, int64_t, int64_t,
                          int64_t, int64_t, int64_t, double);
void fastcomm_gather_cuda(const at::Tensor&, at::Tensor&, int64_t, const at::Tensor&, int64_t, int64_t, int64_t,
                          int64_t, int64_t, double, bool);

// A POSIX shared-memory region mapped here and registered with the current GPU: (host address, device address).
std::vector<int64_t> open_region(const std::string& name, int64_t bytes, bool create) {
    const int fd = shm_open(name.c_str(), create ? (O_CREAT | O_EXCL | O_RDWR) : O_RDWR, 0600);
    TORCH_CHECK(fd >= 0, "fastcomm: shm_open ", name, " failed: ", std::strerror(errno));
    if (create) TORCH_CHECK(ftruncate(fd, bytes) == 0, "fastcomm: ftruncate failed: ", std::strerror(errno));
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    TORCH_CHECK(p != MAP_FAILED, "fastcomm: mmap of ", bytes, " bytes failed: ", std::strerror(errno));
    if (create) std::memset(p, 0, bytes);
    cudaError_t e = cudaHostRegister(p, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
    TORCH_CHECK(e == cudaSuccess, "fastcomm: cudaHostRegister failed: ", cudaGetErrorString(e));
    void* d = nullptr;
    e = cudaHostGetDevicePointer(&d, p, 0);
    TORCH_CHECK(e == cudaSuccess, "fastcomm: cudaHostGetDevicePointer failed: ", cudaGetErrorString(e));
    return {reinterpret_cast<int64_t>(p), reinterpret_cast<int64_t>(d)};
}

void unlink_region(const std::string& name) { shm_unlink(name.c_str()); }

void close_region(int64_t host, int64_t bytes) {
    cudaHostUnregister(reinterpret_cast<void*>(host));
    munmap(reinterpret_cast<void*>(host), bytes);
}

void reduce(const at::Tensor& in, at::Tensor& out, int64_t region, const at::Tensor& epoch, int64_t cap, int64_t rcap,
            int64_t rank, int64_t world, int64_t mode, int64_t blocks, double timeout_s) {
    TORCH_CHECK(epoch.is_cuda() && epoch.scalar_type() == at::kInt && epoch.numel() >= 2, "epoch: 2 int32 on the GPU");
    TORCH_CHECK(world >= 1 && world <= 8 && rank >= 0 && rank < world, "fastcomm: 1 to 8 ranks");
    c10::cuda::CUDAGuard guard(in.device());
    fastcomm_reduce_cuda(in, out, region, epoch, cap, rcap, rank, world, mode, blocks, timeout_s);
}

void gather(const at::Tensor& in, at::Tensor& out, int64_t region, const at::Tensor& epoch, int64_t cap, int64_t rcap,
            int64_t rank, int64_t world, int64_t blocks, double timeout_s, bool ll) {
    TORCH_CHECK(epoch.is_cuda() && epoch.scalar_type() == at::kInt && epoch.numel() >= 2, "epoch: 2 int32 on the GPU");
    TORCH_CHECK(world >= 1 && world <= 8 && rank >= 0 && rank < world, "fastcomm: 1 to 8 ranks");
    c10::cuda::CUDAGuard guard(in.device());
    fastcomm_gather_cuda(in, out, region, epoch, cap, rcap, rank, world, blocks, timeout_s, ll);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("region_bytes", &fastcomm_region_bytes);
    m.def("open_region", &open_region);
    m.def("unlink_region", &unlink_region);
    m.def("close_region", &close_region);
    m.def("reduce", &reduce);
    m.def("gather", &gather);
}
