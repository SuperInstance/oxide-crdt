//! # oxide-crdt
//!
//! GPU-aware CRDT types for distributed state synchronization in the Flux→PTX runtime.
//!
//! When GPU work is distributed across multiple nodes, state must be reconciled
//! without coordination. This crate provides CRDTs specifically designed for:
//!
//! - **Kernel state**: which kernels are loaded, on which GPUs
//! - **Agent assignments**: which agent runs on which GPU
//! - **Metrics**: throughput, latency, errors — merged causally
//! - **Construct registry**: available skills/equipment across the fleet

use std::collections::HashMap;

/// Lamport timestamp for causal ordering.
pub type VectorClock = HashMap<String, u64>;

/// A node in the GPU fleet.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(pub String);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// CRDT operation outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeResult<T> {
    /// Value was updated.
    Updated(T),
    /// Value was already up-to-date (no-op).
    AlreadyCurrent,
    /// Conflict resolved (value changed).
    ConflictResolved(T),
}

/// A G-Counter (grow-only counter) — tracks metrics across nodes.
#[derive(Debug, Clone)]
pub struct GCounter {
    counts: HashMap<String, u64>,
}

impl GCounter {
    pub fn new() -> Self {
        Self { counts: HashMap::new() }
    }

    /// Increment this node's counter.
    pub fn inc(&mut self, node: &str, delta: u64) {
        *self.counts.entry(node.to_string()).or_insert(0) += delta;
    }

    /// Get the total count across all nodes.
    pub fn total(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Merge another G-Counter into this one (takes max per node).
    pub fn merge(&mut self, other: &GCounter) {
        for (node, count) in &other.counts {
            let entry = self.counts.entry(node.clone()).or_insert(0);
            *entry = (*entry).max(*count);
        }
    }
}

impl Default for GCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// A PN-Counter (increment + decrement) — tracks net metrics.
#[derive(Debug, Clone)]
pub struct PNCounter {
    p: GCounter, // increments
    n: GCounter, // decrements
}

impl PNCounter {
    pub fn new() -> Self {
        Self { p: GCounter::new(), n: GCounter::new() }
    }

    pub fn inc(&mut self, node: &str, delta: u64) {
        self.p.inc(node, delta);
    }

    pub fn dec(&mut self, node: &str, delta: u64) {
        self.n.inc(node, delta);
    }

    pub fn value(&self) -> i64 {
        self.p.total() as i64 - self.n.total() as i64
    }

    pub fn merge(&mut self, other: &PNCounter) {
        self.p.merge(&other.p);
        self.n.merge(&other.n);
    }
}

impl Default for PNCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Kernel state on a specific GPU node — LWW-Register.
#[derive(Debug, Clone)]
pub struct KernelState {
    pub kernel_name: String,
    pub node: NodeId,
    pub loaded: bool,
    pub version: u64,
    pub timestamp: u64,
}

/// Last-Write-Wins Register for kernel state.
#[derive(Debug, Clone)]
pub struct LwwKernelMap {
    states: HashMap<String, KernelState>,
}

impl LwwKernelMap {
    pub fn new() -> Self {
        Self { states: HashMap::new() }
    }

    /// Set kernel state on a node.
    pub fn set(&mut self, kernel: &str, node: NodeId, loaded: bool, timestamp: u64) -> MergeResult<bool> {
        let key = format!("{}:{}", kernel, node);
        match self.states.get(&key) {
            Some(existing) if existing.timestamp >= timestamp => MergeResult::AlreadyCurrent,
            _ => {
                self.states.insert(key, KernelState {
                    kernel_name: kernel.to_string(),
                    node,
                    loaded,
                    version: 0,
                    timestamp,
                });
                MergeResult::Updated(loaded)
            }
        }
    }

    /// Check if a kernel is loaded on a specific node.
    pub fn is_loaded(&self, kernel: &str, node: &NodeId) -> bool {
        let key = format!("{}:{}", kernel, node);
        self.states.get(&key).map(|s| s.loaded).unwrap_or(false)
    }

    /// List all nodes where a kernel is loaded.
    pub fn loaded_nodes(&self, kernel: &str) -> Vec<&NodeId> {
        self.states.values()
            .filter(|s| s.kernel_name == kernel && s.loaded)
            .map(|s| &s.node)
            .collect()
    }

    /// Merge another LWW map (last-write-wins per key).
    pub fn merge(&mut self, other: &LwwKernelMap) {
        for (key, state) in &other.states {
            match self.states.get(key) {
                Some(existing) if existing.timestamp >= state.timestamp => {}
                _ => {
                    self.states.insert(key.clone(), state.clone());
                }
            }
        }
    }

    /// Total kernels loaded across all nodes.
    pub fn total_loaded(&self) -> usize {
        self.states.values().filter(|s| s.loaded).count()
    }
}

impl Default for LwwKernelMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Agent assignment — which agent runs on which GPU.
#[derive(Debug, Clone)]
pub struct AgentAssignment {
    pub agent_id: String,
    pub gpu_node: NodeId,
    pub gpu_index: u32,
    pub assigned_at: u64,
}

/// OR-Set (Observed-Remove Set) for agent assignments.
#[derive(Debug, Clone)]
pub struct AgentAssignmentSet {
    assignments: HashMap<String, AgentAssignment>,
    tombstones: HashMap<String, u64>,
}

impl AgentAssignmentSet {
    pub fn new() -> Self {
        Self { assignments: HashMap::new(), tombstones: HashMap::new() }
    }

    /// Assign an agent to a GPU.
    pub fn assign(&mut self, agent_id: &str, node: NodeId, gpu_index: u32, timestamp: u64) {
        if let Some(&ts) = self.tombstones.get(agent_id) {
            if ts >= timestamp { return; }
        }
        self.assignments.insert(agent_id.to_string(), AgentAssignment {
            agent_id: agent_id.to_string(),
            gpu_node: node,
            gpu_index,
            assigned_at: timestamp,
        });
    }

    /// Remove an agent assignment.
    pub fn remove(&mut self, agent_id: &str, timestamp: u64) {
        self.assignments.remove(agent_id);
        self.tombstones.insert(agent_id.to_string(), timestamp);
    }

    /// Get an agent's assignment.
    pub fn get(&self, agent_id: &str) -> Option<&AgentAssignment> {
        self.assignments.get(agent_id)
    }

    /// Merge another assignment set.
    pub fn merge(&mut self, other: &AgentAssignmentSet) {
        for (id, ts) in &other.tombstones {
            let entry = self.tombstones.entry(id.clone()).or_insert(0);
            *entry = (*entry).max(*ts);
        }
        for (id, assignment) in &other.assignments {
            if let Some(&ts) = self.tombstones.get(id) {
                if ts >= assignment.assigned_at { continue; }
            }
            self.assignments.insert(id.clone(), assignment.clone());
        }
    }

    /// Total assigned agents.
    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }
}

impl Default for AgentAssignmentSet {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU metrics CRDT — aggregated performance data across nodes.
#[derive(Debug, Clone)]
pub struct GpuMetricsCrdt {
    /// Operations counter.
    pub ops: PNCounter,
    /// Errors counter.
    pub errors: GCounter,
    /// Total latency in microseconds.
    pub latency_us: GCounter,
    /// Peak memory per node (bytes).
    pub peak_memory: HashMap<String, u64>,
}

impl GpuMetricsCrdt {
    pub fn new() -> Self {
        Self {
            ops: PNCounter::new(),
            errors: GCounter::new(),
            latency_us: GCounter::new(),
            peak_memory: HashMap::new(),
        }
    }

    /// Record an operation.
    pub fn record_op(&mut self, node: &str, latency_us: u64, memory_bytes: u64) {
        self.ops.inc(node, 1);
        self.latency_us.inc(node, latency_us);
        let peak = self.peak_memory.entry(node.to_string()).or_insert(0);
        *peak = (*peak).max(memory_bytes);
    }

    /// Record an error.
    pub fn record_error(&mut self, node: &str) {
        self.errors.inc(node, 1);
    }

    /// Average latency across all operations.
    pub fn avg_latency_us(&self) -> f64 {
        let total_ops = self.ops.value();
        if total_ops == 0 { return 0.0; }
        self.latency_us.total() as f64 / total_ops as f64
    }

    /// Error rate (0.0 to 1.0).
    pub fn error_rate(&self) -> f64 {
        let total_ops = self.ops.p.total();
        if total_ops == 0 { return 0.0; }
        self.errors.total() as f64 / total_ops as f64
    }

    /// Merge metrics from another node.
    pub fn merge(&mut self, other: &GpuMetricsCrdt) {
        self.ops.merge(&other.ops);
        self.errors.merge(&other.errors);
        self.latency_us.merge(&other.latency_us);
        for (node, &mem) in &other.peak_memory {
            let entry = self.peak_memory.entry(node.clone()).or_insert(0);
            *entry = (*entry).max(mem);
        }
    }
}

impl Default for GpuMetricsCrdt {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcounter() {
        let mut c1 = GCounter::new();
        c1.inc("node-a", 10);
        c1.inc("node-b", 20);
        assert_eq!(c1.total(), 30);

        let mut c2 = GCounter::new();
        c2.inc("node-a", 15);
        c2.inc("node-c", 5);

        c1.merge(&c2);
        assert_eq!(c1.total(), 40); // max(10,15) + 20 + 5
    }

    #[test]
    fn test_pncounter() {
        let mut c = PNCounter::new();
        c.inc("node-a", 100);
        c.dec("node-a", 30);
        assert_eq!(c.value(), 70);
    }

    #[test]
    fn test_lww_kernel_map() {
        let mut map = LwwKernelMap::new();
        map.set("attention", NodeId("gpu-1".into()), true, 100);
        map.set("attention", NodeId("gpu-2".into()), true, 101);

        assert!(map.is_loaded("attention", &NodeId("gpu-1".into())));
        assert_eq!(map.loaded_nodes("attention").len(), 2);
        assert_eq!(map.total_loaded(), 2);

        // Unload on gpu-1
        map.set("attention", NodeId("gpu-1".into()), false, 200);
        assert_eq!(map.total_loaded(), 1);
    }

    #[test]
    fn test_lww_merge() {
        let mut m1 = LwwKernelMap::new();
        let mut m2 = LwwKernelMap::new();

        m1.set("reduce", NodeId("gpu-1".into()), true, 100);
        m2.set("reduce", NodeId("gpu-2".into()), true, 100);
        m2.set("reduce", NodeId("gpu-1".into()), false, 150); // newer

        m1.merge(&m2);
        assert!(!m1.is_loaded("reduce", &NodeId("gpu-1".into()))); // updated to false
        assert!(m1.is_loaded("reduce", &NodeId("gpu-2".into())));
    }

    #[test]
    fn test_agent_assignments() {
        let mut set = AgentAssignmentSet::new();
        set.assign("agent-1", NodeId("gpu-1".into()), 0, 100);
        set.assign("agent-2", NodeId("gpu-1".into()), 0, 101);

        assert_eq!(set.len(), 2);
        assert!(set.get("agent-1").is_some());

        set.remove("agent-1", 200);
        assert_eq!(set.len(), 1);
        assert!(set.get("agent-1").is_none());
    }

    #[test]
    fn test_agent_merge() {
        let mut s1 = AgentAssignmentSet::new();
        let mut s2 = AgentAssignmentSet::new();

        s1.assign("agent-1", NodeId("gpu-1".into()), 0, 100);
        s2.assign("agent-2", NodeId("gpu-2".into()), 0, 100);

        s1.merge(&s2);
        assert_eq!(s1.len(), 2);
    }

    #[test]
    fn test_gpu_metrics() {
        let mut m = GpuMetricsCrdt::new();
        m.record_op("gpu-1", 1000, 1024);
        m.record_op("gpu-1", 2000, 2048);
        m.record_error("gpu-1");

        assert_eq!(m.ops.value(), 2);
        assert!((m.avg_latency_us() - 1500.0).abs() < 0.001);
        assert!((m.error_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_metrics_merge() {
        let mut m1 = GpuMetricsCrdt::new();
        let mut m2 = GpuMetricsCrdt::new();

        m1.record_op("gpu-1", 1000, 1024);
        m2.record_op("gpu-2", 500, 512);

        m1.merge(&m2);
        assert_eq!(m1.ops.value(), 2);
        assert_eq!(*m1.peak_memory.get("gpu-1").unwrap(), 1024);
        assert_eq!(*m1.peak_memory.get("gpu-2").unwrap(), 512);
    }
}
