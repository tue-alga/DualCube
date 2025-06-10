// use crate::douconel::{Douconel, EdgeID, FaceID, VertID};
// use crate::douconel_embedded::HasPosition;
// use petgraph::graphmap::DiGraphMap;
// use std::collections::HashSet;

// impl<V: Default, E: Default, F: Default> Douconel<V, E, F> {
//     // Iterator of edges into petgraph graph.
//     pub fn graph_from_edges<I, W, T>(&self, edges: I, weight_function: W) -> DiGraphMap<VertID, T>
//     where
//         I: Iterator<Item = EdgeID>,
//         W: Fn(EdgeID) -> T,
//     {
//         DiGraphMap::<VertID, T>::from_edges(edges.map(|id| {
//             let (u, v) = self.endpoints(id);
//             (u, v, weight_function(id))
//         }))
//     }

//     // To petgraph.
//     pub fn graph<W, T>(&self, weight_function: W) -> DiGraphMap<VertID, T>
//     where
//         W: Fn(EdgeID) -> T,
//     {
//         self.graph_from_edges(self.edges.keys(), weight_function)
//     }

//     // To petgraph.
//     // Filter the edges in `filter_edges`, and the vertices in `filter_verts`.
//     pub fn graph_filtered<W, T>(
//         &self,
//         filter_verts: &HashSet<VertID>,
//         filter_edges: &HashSet<EdgeID>,
//         weight_function: W,
//     ) -> DiGraphMap<VertID, T>
//     where
//         W: Fn(EdgeID) -> T,
//     {
//         self.graph_from_edges(
//             self.edges.keys().filter(|&id| {
//                 !filter_edges.contains(&id)
//                     && !filter_verts.contains(&self.root(id))
//                     && !filter_verts.contains(&self.root(self.twin(id)))
//             }),
//             weight_function,
//         )
//     }

//     // // To petgraph, copy only nodes.
//     // pub fn graph_nodes(&self) -> DiGraphMap<VertID, f32> {
//     //     let mut g = DiGraphMap::<VertID, f32>::new();
//     //     for id in self.verts.keys() {
//     //         g.add_node(id);
//     //     }
//     //     g
//     // }

//     // // To petgraph: dual graph
//     // pub fn dual_graph(&self) -> DiGraphMap<FaceID, ()> {
//     //     let mut edges = vec![];
//     //     for id in self.faces.keys() {
//     //         for n_id in self.fneighbors(id) {
//     //             edges.push((id, n_id, ()));
//     //         }
//     //     }

//     //     DiGraphMap::<FaceID, ()>::from_edges(edges)
//     // }
// }

// impl<V: Default + HasPosition, E: Default, F: Default> Douconel<V, E, F> {
//     // To petgraph, directed primal graph, based on the DCEL, with Euclidean distance as weights.
//     #[must_use]
//     pub fn graph_euclidean(&self) -> DiGraphMap<VertID, f32> {
//         self.graph(|id| self.length(id))
//     }

//     // To petgraph: directed dual graph, based on the dual of the DCEL, with Euclidean distance (face centroid to face centroid) as weights.
//     #[must_use]
//     pub fn dual_graph_euclidean(&self) -> DiGraphMap<FaceID, f32> {
//         let mut edges = vec![];
//         for id in self.faces.keys() {
//             for n_id in self.fneighbors(id) {
//                 edges.push((id, n_id, self.centroid(id).distance(self.centroid(n_id))));
//             }
//         }

//         DiGraphMap::<FaceID, f32>::from_edges(edges)
//     }
// }

// // impl<V: HasPosition, E, F: HasNormal> Douconel<V, E, F> {
// //     // To petgraph: edge graph with <>DWAJD@$@!KM# edge weights
// //     pub fn edge_graph(&self, direction: Vec3, gamma: f32, filter: f32) -> DiGraphMap<EdgeID, f32> {
// //         let mut edges = vec![];
// //         let mut verts = HashSet::new();

// //         for id in self.edges.keys() {
// //             for n_id in self.edges(self.face(id)) {
// //                 if id == n_id {
// //                     continue;
// //                 }

// //                 let edge_direction = (self.midpoint(n_id) - self.midpoint(id)).normalize();
// //                 let edge_normal = self.edge_normal(id);
// //                 let cross = edge_direction.cross(edge_normal);
// //                 let angle = (direction.angle_between(cross) / std::f32::consts::PI) * 180.;
// //                 let weight = angle.powf(gamma);

// //                 if angle < filter {
// //                     edges.push((id, n_id, weight));
// //                     verts.insert(id);
// //                     verts.insert(n_id);
// //                 }
// //             }
// //         }

// //         for id in verts {
// //             let n_id = self.twin(id);

// //             let edge_direction = self.vector(n_id).normalize();
// //             let angle = (direction.angle_between(edge_direction) / std::f32::consts::PI) * 180.;
// //             println!("{:?}", angle);
// //             let weight = angle.powf(gamma);

// //             if angle < filter {
// //                 edges.push((id, n_id, weight));
// //             }
// //         }

// //         DiGraphMap::<EdgeID, f32>::from_edges(edges)
// //     }

// //     // Graph s.t. node for each original edge, and (directed) edges between nodes if the original edges share a face
// //     // Nodes have as position the midpoint of the original edge
// //     #[must_use]
// //     pub fn midpoint_graph(&self) -> DiGraphMap<EdgeID, ()> {
// //         let mut edges = vec![];
// //         for id in self.edges.keys() {
// //             if self.twin(id) < id {
// //                 continue;
// //             }

// //             for n_id in self.edges(self.face(id)) {
// //                 if id == n_id {
// //                     continue;
// //                 }
// //                 edges.push((id, n_id, ()));
// //                 edges.push((n_id, id, ()));
// //             }
// //             for n_id in self.edges(self.face(self.twin(id))) {
// //                 if self.twin(id) == n_id {
// //                     continue;
// //                 }
// //                 edges.push((id, n_id, ()));
// //                 edges.push((n_id, id, ()));
// //             }
// //         }
// //         DiGraphMap::<EdgeID, ()>::from_edges(edges)
// //     }
// // }

// // impl<V: HasPosition, E, F: HasNormal> Douconel<V, E, F> {
// //     // To petgraph: edge graph with <>DWAJD@$@!KM# edge weights
// //     pub fn special_orientation_graph(
// //         &self,
// //         direction: Vec3,
// //         gamma: f32,
// //     ) -> DiGraphMap<EdgeID, f32> {
// //         let mut orientations = HashSet::new();

// //         let mut prio: PriorityQueue<f32, EdgeID> = PriorityQueue::from(
// //             self.edges
// //                 .keys()
// //                 .map(|edge_id| {
// //                     let edge_direction = self.vector(edge_id).normalize();
// //                     let angle =
// //                         (edge_direction.angle_between(direction) / std::f32::consts::PI) * 180.;

// //                     (angle, edge_id)
// //                 })
// //                 .collect::<Vec<_>>(),
// //         );

// //         while let Some((_, edge_id)) = prio.pop() {
// //             let twin_id = self.twin(edge_id);
// //             if !orientations.contains(&twin_id) {
// //                 orientations.insert(edge_id);
// //             }
// //         }

// //         let mut edges = vec![];
// //         let mut verts = HashSet::new();

// //         for edge_id in self.edges.keys() {
// //             for next_id in self.edges(self.face(edge_id)) {
// //                 if edge_id == next_id {
// //                     continue;
// //                 }

// //                 // Only add an edge, if its orientation is aligned. Which means, given the orientation, the two edges should share either their start or end vertex.
// //                 let (mut edge_root, mut edge_toor) = self.endpoints(edge_id);
// //                 let (mut next_root, mut next_toor) = self.endpoints(next_id);
// //                 if !orientations.contains(&edge_id) {
// //                     // swap the endpoints
// //                     std::mem::swap(&mut edge_root, &mut edge_toor);
// //                 }
// //                 if !orientations.contains(&next_id) {
// //                     // swap the endpoints
// //                     std::mem::swap(&mut next_root, &mut next_toor);
// //                 }

// //                 if edge_root != next_root && edge_toor != next_toor {
// //                     continue;
// //                 }

// //                 let edge_direction = (self.midpoint(next_id) - self.midpoint(edge_id)).normalize();
// //                 let edge_normal = self.edge_normal(edge_id);
// //                 let cross = edge_direction.cross(edge_normal);
// //                 let angle = (direction.angle_between(cross) / std::f32::consts::PI) * 180.;
// //                 let weight = angle.powf(gamma);

// //                 if angle < 90. {
// //                     edges.push((edge_id, next_id, weight));
// //                     verts.insert(edge_id);
// //                     verts.insert(next_id);
// //                 }
// //             }
// //         }

// //         // add edge between twins for connectivity
// //         for edge_id in verts {
// //             let twin_id = self.twin(edge_id);

// //             let edge_direction = self.vector(twin_id).normalize();
// //             let angle = (direction.angle_between(edge_direction) / std::f32::consts::PI) * 180.;
// //             let weight = angle.powf(gamma);

// //             if angle < 90. {
// //                 edges.push((edge_id, twin_id, weight));
// //             }
// //         }

// //         DiGraphMap::<EdgeID, f32>::from_edges(edges)
// //     }
// // }
