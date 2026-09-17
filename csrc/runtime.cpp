#include "runtime.hpp"

#include <cstdint>
#include <optional>

namespace ultra_ep::runtime {

bool is_runtime_initialized = false;

int rank_idx = -1, nvl_rank_idx = -1, rdma_rank_idx = -1;
int num_ranks = 0, num_nvl_ranks = 0, num_rdma_ranks = 0;
int device_id = -1, num_device_sms = 0;

at::cuda::CUDAStream get_global_comm_stream() {
    static std::optional<at::cuda::CUDAStream> comm_stream = std::nullopt;
    if (not comm_stream.has_value())
        comm_stream = at::cuda::getStreamFromPool(true);
    return comm_stream.value();
}

void init_runtime(const int& rank_idx_,
                  const int& num_ranks_, const int& num_nvl_ranks_) {
    EP_HOST_ASSERT(rank_idx_ >= 0 && rank_idx_ < num_ranks_);
    EP_HOST_ASSERT(num_ranks_ > 0 && num_nvl_ranks_ > 0);
    EP_HOST_ASSERT(num_nvl_ranks_ <= kernels::kMaxNvlDomainSize);
    EP_HOST_ASSERT(num_ranks_ % num_nvl_ranks_ == 0);
    num_ranks = num_ranks_;
    num_nvl_ranks = num_nvl_ranks_;
    num_rdma_ranks = num_ranks / num_nvl_ranks;
    rank_idx = rank_idx_;
    nvl_rank_idx = rank_idx % num_nvl_ranks;
    rdma_rank_idx = rank_idx / num_nvl_ranks;

    // Get device info
    CUDA_RUNTIME_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp device_prop = {};
    CUDA_RUNTIME_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
    num_device_sms = device_prop.multiProcessorCount;

    // Available to create buffers
    is_runtime_initialized = true;
}

void destroy() {
    EP_HOST_ASSERT(is_runtime_initialized);

    // Cannot use anymore
    rank_idx = nvl_rank_idx = rdma_rank_idx = -1;
    num_ranks = num_nvl_ranks = num_rdma_ranks = 0;
    is_runtime_initialized = false;
}

void register_apis(pybind11::module_& m) {
    m.def("is_runtime_initialized", []() { return is_runtime_initialized; });
    m.def("init_runtime", &init_runtime);
    m.def("destroy_runtime", &destroy);
}

}  // namespace ultra_ep::runtime
