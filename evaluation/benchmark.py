"""
Benchmark utilities for PixelSentinel.

This module measures model performance characteristics:

    - Inference latency
    - Throughput (images per second)
    - GPU memory consumption
    - Parameter count
    - Model size estimation

No training logic is included.
"""

from __future__ import annotations

import logging
import time

import torch
from torch import Tensor, nn

LOGGER = logging.getLogger(__name__)

__all__ = [
    "count_parameters",
    "estimate_model_size",
    "measure_latency",
    "measure_throughput",
    "benchmark_model",
]


def count_parameters(
    model: nn.Module,
) -> int:
    """
    Count trainable parameters.

    Parameters
    ----------
    model:
        Neural network model.

    Returns
    -------
    int
        Number of trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def estimate_model_size(
    model: nn.Module,
) -> float:
    """
    Estimate model size in megabytes.

    Parameters
    ----------
    model:
        Neural network model.

    Returns
    -------
    float
        Model size in MB.
    """

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    bytes_size = (
        parameters
        * 4
    )

    return bytes_size / (
        1024 ** 2
    )


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    input_tensor: Tensor,
    device: torch.device,
    iterations: int = 100,
) -> float:
    """
    Measure average inference latency.

    Parameters
    ----------
    model:
        Model to benchmark.

    input_tensor:
        Input tensor.

    device:
        Execution device.

    iterations:
        Number of inference iterations.

    Returns
    -------
    float
        Average latency in milliseconds.
    """

    model.eval()

    input_tensor = input_tensor.to(
        device,
        non_blocking=True,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(iterations):

        _ = model(
            input_tensor
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    latency = (
        end - start
    ) / iterations

    return latency * 1000


@torch.no_grad()
def measure_throughput(
    model: nn.Module,
    input_tensor: Tensor,
    device: torch.device,
    duration: float = 10.0,
) -> float:
    """
    Measure inference throughput.

    Parameters
    ----------
    model:
        Model to benchmark.

    input_tensor:
        Input tensor.

    device:
        Execution device.

    duration:
        Benchmark duration in seconds.

    Returns
    -------
    float
        Images processed per second.
    """

    model.eval()

    input_tensor = input_tensor.to(
        device,
        non_blocking=True,
    )

    count = 0

    start = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.synchronize()

    while (
        time.perf_counter() - start
        <
        duration
    ):

        _ = model(
            input_tensor
        )

        count += input_tensor.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        -
        start
    )

    return count / elapsed


@torch.no_grad()
def benchmark_model(
    model: nn.Module,
    input_tensor: Tensor,
    device: torch.device,
) -> dict[str, float]:
    """
    Run complete model benchmark.

    Parameters
    ----------
    model:
        Neural network model.

    input_tensor:
        Example input tensor.

    device:
        Execution device.

    Returns
    -------
    dict[str, float]
        Benchmark results.
    """

    parameters = count_parameters(
        model
    )

    size_mb = estimate_model_size(
        model
    )

    latency = measure_latency(
        model=model,
        input_tensor=input_tensor,
        device=device,
    )

    throughput = measure_throughput(
        model=model,
        input_tensor=input_tensor,
        device=device,
    )

    gpu_memory = 0.0

    if device.type == "cuda":

        gpu_memory = (
            torch.cuda.max_memory_allocated(
                device=device
            )
            /
            (1024 ** 3)
        )

    results = {
        "parameters": float(
            parameters
        ),
        "model_size_mb": size_mb,
        "latency_ms": latency,
        "throughput_images_per_second": (
            throughput
        ),
        "gpu_memory_gb": gpu_memory,
    }

    LOGGER.info(
        "Benchmark results: %s",
        results,
    )

    return results