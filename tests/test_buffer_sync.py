"""Exercise cross-PE buffer lifetimes without barriers around individual calls.

Run on at least two NVLink-connected SM90/SM100 GPUs from the repository root:
    PYTHONPATH=. torchrun --standalone --nproc_per_node=2 tests/test_buffer_sync.py
    PYTHONPATH=. torchrun --standalone --nproc_per_node=2 tests/test_buffer_sync.py --graph

Repeat with --sync and --non-deterministic. On larger domains, also test
--plan-mode force_relay in eager mode. MAX_NUM_NVL_PEERS can partition a local
NVLink domain to exercise multiple team creation, e.g. 4 GPUs with a value of 2.
"""

import argparse
import os

import torch
import torch.distributed as dist

import ultra_ep


def install_placement(manager, layer):
    """One hot master per domain, replicated once on every other PE.

    This fixed fixture isolates communication from placement solving. It also
    guarantees PEs with no outgoing weight copies or local reduction tasks.
    """
    world, domain = manager.num_ranks, manager.nvl_domain_size
    p2l = torch.full((world * 2,), -1, dtype=torch.int32)
    l2p = torch.full((world, world), -1, dtype=torch.int32)
    counts = torch.ones(world, dtype=torch.int32)
    for rank in range(world):
        p2l[rank * 2] = rank
        l2p[rank, 0] = rank * 2
    for base in range(0, world, domain):
        hot = base + layer % domain
        replicas = [rank for rank in range(base, base + domain) if rank != hot]
        counts[hot] = domain
        for slot, rank in enumerate(replicas, start=1):
            p2l[rank * 2 + 1] = hot
            l2p[hot, slot] = rank * 2 + 1
    manager.physical_to_logical_map[layer].copy_(p2l)
    manager.logical_to_physical_map[layer].copy_(l2p)
    manager.logical_replica_counts[layer].copy_(counts)


def run_case(manager, weights, grads, args, operation):
    rank, domain = manager.rank, manager.nvl_domain_size
    local_rank = rank % domain
    domain_base = rank - local_rank
    # Keep validation on-device: a host assertion per call could hide skew.
    errors = torch.zeros((), dtype=torch.int64, device="cuda")
    manager.local_replica_weight_buffer.fill_(-1)
    manager.local_replica_grad_buffer.zero_()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def check(tensor, expected):
        errors.add_(torch.count_nonzero(tensor != expected))

    def batch():
        for step in range(8):
            layer = step % 2
            hot = domain_base + layer
            # Rotate the delayed PE, exercising both remote producers and
            # consumers. This is GPU work, so it also applies during replay.
            if local_rank == (step // 2) % domain:
                torch.cuda._sleep(args.skew_cycles)
            value = step + 1
            if operation == "weight_sync":
                for shard, weight in enumerate(weights[layer]):
                    weight.fill_(rank + 1 + value * manager.num_ranks + shard)
            else:
                for grad in grads[layer]:
                    grad.fill_(value)
                # Wgrad accumulates into storage cleared remotely by the
                # preceding call, rather than overwriting stale values.
                if rank != hot:
                    manager.local_replica_grad_buffer.add_(value)

            event = getattr(manager, operation)(layer, async_finish=not args.sync)
            if not args.sync:
                event.current_stream_wait()

            if operation == "weight_sync":
                # Delay a replica consumer while faster peers advance to the
                # next layer and try to overwrite the same weight allocation.
                if local_rank == (step + 1) % domain:
                    torch.cuda._sleep(args.skew_cycles)
                if rank != hot:
                    check(
                        manager.local_replica_fc1_weight_buffer,
                        hot + 1 + value * manager.num_ranks,
                    )
                    check(
                        manager.local_replica_fc2_weight_buffer,
                        hot + 2 + value * manager.num_ranks,
                    )
            else:
                for grad in grads[layer]:
                    check(grad, value * (domain if rank == hot else 1))
                if rank != hot:
                    check(manager.local_replica_grad_buffer, 0)

    # Setup/warmup boundaries are outside the measured sequences. No external
    # collective or device synchronization separates operations within a batch.
    torch.cuda.synchronize()
    dist.barrier()
    with torch.cuda.stream(stream):
        batch()
    torch.cuda.synchronize()
    dist.barrier()
    if args.graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            batch()
        for _ in range(args.iterations):
            graph.replay()
    else:
        with torch.cuda.stream(stream):
            for _ in range(args.iterations):
                batch()
    torch.cuda.synchronize()
    dist.all_reduce(errors)
    assert errors.item() == 0, f"{operation}: {errors.item()} incorrect elements"
    if rank == 0:
        print(f"PASS {operation}, graph={args.graph}, sync={args.sync}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument(
        "--operation", choices=("both", "weight_sync", "grad_reduce"), default="both"
    )
    parser.add_argument(
        "--plan-mode", choices=("direct", "force_relay"), default="direct"
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--skew-cycles", type=int, default=5_000_000)
    args = parser.parse_args()
    if args.iterations < 1 or args.skew_cycles < 1:
        parser.error("iterations and skew-cycles must be positive")
    if args.graph and args.plan_mode != "direct":
        parser.error("this regression covers CUDA graph replay in direct mode")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    manager = ultra_ep.Manager(
        group=dist.group.WORLD,
        num_layers=2,
        num_local_master_experts=1,
        num_local_redundant_experts=1,
        expert_fc1_numel=262144,
        expert_fc2_numel=131072,
        weight_data_dtype=torch.float32,
        explicitly_destroy=True,
    )
    try:
        assert manager.nvl_domain_size >= 2, (
            "requires at least two NVLink peers per domain"
        )
        if args.plan_mode == "force_relay":
            assert manager.nvl_domain_size >= 4, (
                "relay testing requires at least four NVLink peers"
            )
        manager.set_weight_sync_plan_mode(args.plan_mode)
        manager.set_grad_reduce_deterministic(not args.non_deterministic)
        weights, grads = [], []
        for layer in range(2):
            install_placement(manager, layer)
            weights.append([torch.zeros(n, device="cuda") for n in (262144, 131072)])
            grads.append([torch.zeros_like(w) for w in weights[layer]])
            manager.construct_local_master_ptr_pool(
                layer,
                [weights[layer][0]],
                [weights[layer][1]],
                [grads[layer][0]],
                [grads[layer][1]],
            )
        for operation in ("weight_sync", "grad_reduce"):
            if args.operation in ("both", operation):
                run_case(manager, weights, grads, args, operation)
    finally:
        manager.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
