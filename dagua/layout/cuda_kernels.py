"""Optional CUDA kernels for performance-critical operations.

These kernels are compiled at first use via
``torch.utils.cpp_extension.load_inline`` and remain optional when CUDA is
unavailable.
"""

from __future__ import annotations

from typing import Any

import torch

_csr_module: Any | None = None


def _load_csr_kernel() -> Any:
    """Lazily compile and return the CSR scatter kernel module.

    Returns
    -------
    Any
        Loaded inline extension module exposing ``csr_scatter``.
    """
    global _csr_module
    if _csr_module is not None:
        return _csr_module

    from torch.utils.cpp_extension import load_inline

    cuda_src = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>
    #include <cstdint>

    __global__ void csr_scatter_kernel(
        const int64_t* __restrict__ src,
        const int64_t* __restrict__ tgt,
        int64_t* __restrict__ write_pos,
        int64_t* __restrict__ csr_targets,
        int64_t num_edges
    ) {
        int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= num_edges) return;
        int64_t s = src[idx];
        int64_t pos = static_cast<int64_t>(
            atomicAdd(reinterpret_cast<unsigned long long*>(&write_pos[s]), 1ULL)
        );
        csr_targets[pos] = tgt[idx];
    }

    __global__ void csr_scatter_kernel_int32(
        const int32_t* __restrict__ src,
        const int32_t* __restrict__ tgt,
        int64_t* __restrict__ write_pos,
        int32_t* __restrict__ csr_targets,
        int64_t num_edges
    ) {
        int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= num_edges) return;
        int32_t s = src[idx];
        int64_t pos = static_cast<int64_t>(
            atomicAdd(reinterpret_cast<unsigned long long*>(&write_pos[s]), 1ULL)
        );
        csr_targets[pos] = tgt[idx];
    }

    torch::Tensor csr_scatter(
        torch::Tensor src,
        torch::Tensor tgt,
        torch::Tensor write_pos,
        int64_t num_edges
    ) {
        auto csr_targets = torch::empty({num_edges}, tgt.options());
        if (num_edges == 0) {
            return csr_targets;
        }

        int threads = 256;
        int blocks = static_cast<int>((num_edges + threads - 1) / threads);

        if (src.scalar_type() == torch::kInt32) {
            csr_scatter_kernel_int32<<<blocks, threads>>>(
                src.data_ptr<int32_t>(),
                tgt.data_ptr<int32_t>(),
                write_pos.data_ptr<int64_t>(),
                csr_targets.data_ptr<int32_t>(),
                num_edges
            );
        } else {
            csr_scatter_kernel<<<blocks, threads>>>(
                src.data_ptr<int64_t>(),
                tgt.data_ptr<int64_t>(),
                write_pos.data_ptr<int64_t>(),
                csr_targets.data_ptr<int64_t>(),
                num_edges
            );
        }

        return csr_targets;
    }
    """

    cpp_src = """
    #include <torch/extension.h>

    torch::Tensor csr_scatter(
        torch::Tensor src,
        torch::Tensor tgt,
        torch::Tensor write_pos,
        int64_t num_edges
    );
    """

    _csr_module = load_inline(
        name="dagua_csr_cuda",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=["csr_scatter"],
        verbose=False,
    )
    return _csr_module


def build_csr_cuda(
    src: torch.Tensor,
    tgt: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build CSR adjacency using CUDA atomics in ``O(V + E)`` time.

    Parameters
    ----------
    src : torch.Tensor
        Source node indices on a CUDA device with shape ``(E,)``.
    tgt : torch.Tensor
        Target node indices on the same CUDA device with shape ``(E,)``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(csr_offsets, csr_targets)`` on the same device as ``src`` and
        ``tgt``.
    """
    if src.device.type != "cuda" or tgt.device.type != "cuda":
        raise ValueError("build_csr_cuda requires CUDA tensors")

    edge_count = int(src.shape[0])
    device = src.device

    src_index = src.contiguous()
    tgt_index = tgt.contiguous()

    out_degree = torch.bincount(src_index.to(torch.long), minlength=num_nodes)
    csr_offsets = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    csr_offsets[1:] = out_degree.cumsum(0)

    if edge_count == 0:
        empty_targets = torch.empty(0, dtype=tgt_index.dtype, device=device)
        return csr_offsets, empty_targets

    write_pos = csr_offsets[:-1].clone()
    module = _load_csr_kernel()
    csr_targets = module.csr_scatter(
        src_index,
        tgt_index,
        write_pos,
        edge_count,
    )
    return csr_offsets, csr_targets


def is_available() -> bool:
    """Return whether CUDA CSR kernel compilation is currently available.

    Returns
    -------
    bool
        ``True`` when CUDA is available and the inline extension compiles.
    """
    if not torch.cuda.is_available():
        return False
    try:
        _load_csr_kernel()
        return True
    except Exception:
        return False
