# oxide-crdt

GPU-aware CRDT types for distributed state synchronization in the Flux→PTX runtime.

When you're running kernels across a fleet of GPU nodes, the last thing you want is a consensus round-trip every time an agent migrates or a kernel gets hot-swapped. Network partitions happen. Clocks drift. Nodes reboot. This crate gives you *convergent* data structures that merge correctly without coordination — so your distributed GPU state stays consistent even when the fabric doesn't.

## Why CRDTs for GPU State?

In a multi-node GPU cluster, state lives in too many places at once:

- **Kernel registries** — which PTX modules are loaded on which device
- **Agent assignments** — which agent owns which GPU slice
- **Performance metrics** — ops/sec, latency histograms, error counts, peak memory

Traditional consensus (Raft, Paxos) works, but it's expensive across PCIe/NVLink/InfiniBand boundaries. CRDTs let each node make progress independently and reconcile later. The math guarantees that any two nodes who have seen the same set of updates will arrive at the same state, regardless of order.

This crate is specifically tuned for the GPU-fleet case: timestamps are wall-clock (µs), node IDs map to physical GPUs, and merge operations are designed to be called from the cudaclaw runtime's warp-level consensus layer.

## CRDT Types

### `GCounter` — Grow-Only Counter

The simplest building block. Each node tracks its own increments; merge takes the maximum per node. Perfect for "how many total ops did we process?" where ops never need to roll back.

```rust
use oxide_crdt::GCounter;

let mut ops = GCounter::new();
ops.inc("gpu-node-1", 100);
ops.inc("gpu-node-2", 250);

assert_eq!(ops.total(), 350); // 100 + 250
```

Merge semantics: `max(counts[node])` across replicas. Monotonic, commutative, associative.

### `PNCounter` — Increment/Decrement Counter

Two `GCounter`s under the hood: one for increments (`P`), one for decrements (`N`). Net value is `P - N`. Use this when you need signed quantities — queue depth, active slot count, budget balances.

```rust
use oxide_crdt::PNCounter;

let mut slots = PNCounter::new();
slots.inc("gpu-node-1", 10); // 10 slots allocated
slots.dec("gpu-node-1", 3);  // 3 freed

assert_eq!(slots.value(), 7);
```

Merge semantics: merge `P` and `N` independently as `GCounter`s. Value converges to the globally correct net count.

### `LwwKernelMap` — Last-Write-Wins Kernel Registry

Maps `(kernel_name, node_id)` to a `KernelState`. The timestamp field resolves conflicts: whichever write carries the later timestamp wins. This is how we track which kernels are loaded where, without a central registry.

```rust
use oxide_crdt::{LwwKernelMap, NodeId};

let mut kernels = LwwKernelMap::new();

// Load attention kernel on gpu-1 at t=100
kernels.set("attention", NodeId("gpu-1".into()), true, 100);

// Load it on gpu-2 at t=101
kernels.set("attention", NodeId("gpu-2".into()), true, 101);

assert!(kernels.is_loaded("attention", &NodeId("gpu-1".into())));
assert_eq!(kernels.loaded_nodes("attention").len(), 2);

// Hot-unload on gpu-1 at t=200
kernels.set("attention", NodeId("gpu-1".into()), false, 200);
assert_eq!(kernels.total_loaded(), 1);
```

Key design decision: each `(kernel, node)` pair is an independent LWW register. This means `reduce` on `gpu-1` and `reduce` on `gpu-2` never contend with each other — they're different keys. The merge walks the other replica's map and keeps the entry with the higher timestamp.

**Conflict resolution:** If two nodes write the same key with the same timestamp, the last writer in the merge wins (deterministic by iteration order). In practice, use microsecond timestamps from a monotonic source and collisions are negligible.

### `AgentAssignmentSet` — OR-Set for Agent Placement

An Observed-Remove Set tracks which agents are assigned to which GPU. The twist: removal is *tombstoned*. If node A removes agent X at t=200, and a stale message from node B adds X at t=150, the tombstone wins. Without this, you'd see "ghost" re-insertions after removal.

```rust
use oxide_crdt::{AgentAssignmentSet, NodeId};

let mut fleet = AgentAssignmentSet::new();

fleet.assign("agent-7", NodeId("gpu-1".into()), 0, 100);
fleet.assign("agent-8", NodeId("gpu-2".into()), 1, 101);

assert_eq!(fleet.len(), 2);

// Agent-7 moves off gpu-1
fleet.remove("agent-7", 200);
assert!(fleet.get("agent-7").is_none());

// Stale add from a partitioned node at t=150 — correctly ignored
fleet.assign("agent-7", NodeId("gpu-3".into()), 0, 150);
assert!(fleet.get("agent-7").is_none());
```

Merge semantics:
1. Tombstones merge with `max(timestamp)`.
2. An assignment is kept only if its timestamp is strictly greater than the tombstone for that agent.

This gives you exactly-once placement semantics without a coordinator.

### `GpuMetricsCrdt` — Aggregated Fleet Metrics

The top-level metrics structure. Composes `PNCounter` for ops, `GCounter` for errors and cumulative latency, and a `HashMap` for per-node peak memory. This is what you export to your observability stack.

```rust
use oxide_crdt::GpuMetricsCrdt;

let mut metrics = GpuMetricsCrdt::new();

// Record two ops on gpu-1
metrics.record_op("gpu-1", 1_000, 1024); // 1ms latency, 1KiB
metrics.record_op("gpu-1", 2_000, 2048); // 2ms latency, 2KiB

// One error
metrics.record_error("gpu-1");

assert_eq!(metrics.ops.value(), 2);
assert_eq!(metrics.avg_latency_us(), 1500.0);
assert_eq!(metrics.error_rate(), 0.5);
assert_eq!(*metrics.peak_memory.get("gpu-1").unwrap(), 2048);
```

Merge semantics: ops, errors, and latency merge as counters; peak memory takes the `max` per node. After merging two replicas, `avg_latency_us()` and `error_rate()` reflect the global fleet, not just the local view.

## Merge Semantics and Conflict Resolution

Every type in this crate implements a `merge(&mut self, other: &Self)` that is:

- **Commutative:** `a.merge(b)` yields the same state as `b.merge(a)`
- **Associative:** `(a.merge(b)).merge(c)` == `a.merge(b.merge(c))`
- **Idempotent:** `a.merge(b)` is safe to repeat; no double-counting

These aren't just nice properties — they're what let you broadcast deltas over gossip, buffer them in queues, and retry on packet loss without corruption.

| Type | Conflict Strategy | Correctness Guarantee |
|------|-------------------|----------------------|
| `GCounter` | `max` per node | Monotonic; never loses increments |
| `PNCounter` | `max` on both `P` and `N` | Net value converges globally |
| `LwwKernelMap` | Higher timestamp wins | Last write dominates; older values discarded |
| `AgentAssignmentSet` | Tombstone timestamp wins | Removed agents stay removed |
| `GpuMetricsCrdt` | Composed merges + `max` memory | Global metrics converge causally |

## Relationship to SmartCRDT and cudaclaw

**oxide-crdt** sits in the middle of the stack:

- **SmartCRDT** (TypeScript monorepo) handles the fleet-level orchestration: agent discovery, vector search, Docker composition, and the web-facing APIs. When a SmartCRDT node needs to sync kernel state with a GPU-native peer, it speaks the same CRDT algebra — but this crate is the ground-truth implementation for the Rust side.

- **cudaclaw** is the persistent CUDA kernel runtime. It loads PTX, manages warp-level consensus, and pushes 400K+ ops/sec. cudaclaw doesn't know about CRDT theory; it just calls `merge()` on the state buffers that oxide-crdt produces. The cudaclaw-bridge crate (Rust) maps between oxide-crdt's data structures and cudaclaw's device memory layout.

In other words: SmartCRDT decides *what* to run, cudaclaw runs it, and oxide-crdt makes sure both agree on the current state without a round-trip to a database.

## When to Use What

| Problem | Reach for |
|---------|-----------|
| "How many kernels launched total?" | `GCounter` |
| "What's the net queue depth?" | `PNCounter` |
| "Is `transformer-v3` loaded on `gpu-4`?" | `LwwKernelMap` |
| "Where did agent-42 get assigned?" | `AgentAssignmentSet` |
| "What's the fleet-wide p99 latency?" | `GpuMetricsCrdt` |

## Cargo

```toml
[dependencies]
oxide-crdt = "0.1.0"
```

No external dependencies. Pure stdlib. Compiles to WASM if you need it in the browser dashboard.

## License

Apache-2.0
