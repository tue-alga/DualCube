use itertools::Itertools;
use ordered_float::{FloatCore, OrderedFloat};
use petgraph::algo::{astar, tarjan_scc, Measure};
use petgraph::{graph::NodeIndex, Directed, Graph};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;

// Graph struct, that builds an underlying Petgraph with helper functions for various graph algorithms, such as, shortest path, shortest cycle, connected components, etc.
// Also contains functionality to transform a graph into a modified graph. (e.g., filtering edges or vertices)
#[derive(Default, Clone, Serialize, Deserialize, Debug)]
pub struct Graaf<V: Eq + PartialEq + Hash, E> {
    petgraph: Graph<V, E, Directed>,
    node_to_index: HashMap<V, NodeIndex>,
    nodes: Vec<V>,
    edges: Vec<(V, V, E)>,
}

impl<V: Eq + PartialEq + Hash + Default + Copy, E: Copy> Graaf<V, E> {
    pub fn from(nodes: Vec<V>, edges: Vec<(V, V, E)>) -> Self {
        let mut petgraph = Graph::with_capacity(nodes.len(), edges.len());
        let node_to_index: HashMap<V, NodeIndex> = nodes.iter().map(|&node| (node, petgraph.add_node(node))).collect();
        let edges_indexed = edges.iter().map(|(from, to, w)| (node_to_index[from], node_to_index[to], w));
        petgraph.extend_with_edges(edges_indexed);
        Self {
            petgraph,
            node_to_index,
            nodes,
            edges,
        }
    }

    pub fn nodes(&self) -> Vec<V> {
        self.nodes.clone()
    }

    pub fn edges(&self) -> Vec<(V, V, E)> {
        self.edges.clone()
    }

    pub fn filter_edges(&self, predicate: impl Fn((&V, &V)) -> bool) -> Self {
        let nodes = self.nodes.clone();
        let edges = self.edges.iter().filter(|(from, to, _)| predicate((from, to))).copied().collect_vec();
        Self::from(nodes, edges)
    }

    pub fn filter_nodes(&self, predicate: impl Fn(&V) -> bool) -> Self {
        let nodes = self.nodes.iter().filter(|&&node| predicate(&node)).copied().collect_vec();
        let edges = self
            .edges
            .iter()
            .filter(|(from, to, _)| predicate(from) && predicate(to))
            .copied()
            .collect_vec();
        Self::from(nodes, edges)
    }

    pub fn extend(&mut self, nodes: Vec<V>, edges: Vec<(V, V, E)>) {
        let extra_node_to_index: HashMap<V, NodeIndex> = nodes.iter().map(|&node| (node, self.petgraph.add_node(node))).collect();
        self.node_to_index.extend(extra_node_to_index);

        let extra_edges_indexed = edges.iter().map(|(from, to, w)| (self.node_to_index[from], self.node_to_index[to], w));
        self.petgraph.extend_with_edges(extra_edges_indexed);

        self.edges.append(&mut edges.clone());
    }

    pub fn node_to_index(&self, node: &V) -> Option<NodeIndex> {
        self.node_to_index.get(node).copied()
    }

    pub fn index_to_node(&self, index: NodeIndex) -> Option<&V> {
        self.petgraph.node_weight(index)
    }

    pub fn directed_edge_exists(&self, a: V, b: V) -> bool {
        self.neighbors(a).iter().any(|n| n == &b)
    }

    pub fn node_exists(&self, a: V) -> bool {
        self.node_to_index.contains_key(&a)
    }

    pub fn edge_exists(&self, a: V, b: V) -> bool {
        self.directed_edge_exists(a, b) || self.directed_edge_exists(b, a)
    }

    pub fn shortest_path<W: Measure + Copy, F: Fn(E) -> W>(&self, a: NodeIndex, b: NodeIndex, measure: &F) -> Option<(W, Vec<NodeIndex>)> {
        astar(&self.petgraph, a, |finish| finish == b, |e| measure(e.weight().to_owned()), |_| W::default())
    }

    pub fn shortest_path_with_approx<W: Measure + Copy, F: Fn(E) -> W, F2: Fn(V, V) -> W>(
        &self,
        a: NodeIndex,
        b: NodeIndex,
        measure: &F,
        approx: &F2,
    ) -> Option<(W, Vec<NodeIndex>)> {
        astar(
            &self.petgraph,
            a,
            |finish| finish == b,
            |e| measure(e.weight().to_owned()),
            |v| {
                approx(
                    self.petgraph.node_weight(v).unwrap().to_owned(),
                    self.petgraph.node_weight(b).unwrap().to_owned(),
                )
            },
        )
    }

    pub fn neighbors(&self, a: V) -> Vec<V> {
        self.petgraph
            .neighbors(self.node_to_index[&a])
            .map(|index| self.index_to_node(index).unwrap().to_owned())
            .collect()
    }

    pub fn neighbors_undirected(&self, a: V) -> Vec<V> {
        self.petgraph
            .neighbors_undirected(self.node_to_index[&a])
            .map(|index| self.index_to_node(index).unwrap().to_owned())
            .collect()
    }

    pub fn shortest_cycle<W: Measure + Copy + FloatCore, F: Fn(E) -> W>(&self, a: NodeIndex, measure: &F) -> Option<Vec<NodeIndex>> {
        self.petgraph
            .neighbors(a)
            .map(|b| (a, b))
            .filter_map(|(a, b)| {
                let extra = measure(self.get_weight(a, b));
                let path = self.shortest_path(b, a, measure);
                path.map(|(cost, path)| (path, cost + extra))
            })
            .min_by_key(|(_, cost)| OrderedFloat(cost.to_owned()))
            .map(|(path, _)| path)
    }

    pub fn shortest_cycle_edge<W: Measure + Copy + FloatCore, F: Fn(E) -> W>(
        &self,
        (a, b): (NodeIndex, NodeIndex),
        measure: &F,
    ) -> Option<(Vec<NodeIndex>, W)> {
        let path = self.shortest_path(b, a, measure);
        path.map(|(cost, path)| (path, cost))
    }

    pub fn get_weight(&self, a: NodeIndex, b: NodeIndex) -> E {
        self.petgraph.edges_connecting(a, b).next().unwrap().weight().to_owned()
    }

    pub fn cc(&self) -> Vec<Vec<V>> {
        tarjan_scc(&self.petgraph)
            .into_iter()
            .map(|cc| cc.into_iter().map(|index| self.index_to_node(index).unwrap().to_owned()).collect_vec())
            .collect_vec()
    }
}
