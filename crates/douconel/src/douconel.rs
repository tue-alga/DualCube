use bimap::BiHashMap;
use core::panic;
use itertools::Itertools;
use rand::seq::IteratorRandom;
use serde::{Deserialize, Serialize};
use slotmap::Key;
use slotmap::SecondaryMap;
use slotmap::SlotMap;
use std::collections::HashMap;
use std::fmt::Debug;
use thiserror::Error;

#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum MeshError<VertID> {
    #[error("({0}, {1}) does not have a twin (mesh is not a closed 2-manifold)")]
    NoTwin(VertID, VertID),
    #[error("({0}, {1}) exists multiple times (mesh is not a closed 2-manifold)")]
    DuplicateEdge(VertID, VertID),
    #[error("Mesh is not orientable")]
    NotOrientable,
    #[error("Mesh is not connected")]
    NotConnected,
    #[error("Unknown error ({0})")]
    Unknown(String),
}

pub type Empty = u8;

// This is a struct that defines a mesh with vertices, edges, and faces.
// This mesh is:
//      a closed 2-manifold: Each edge corresponds to exactly two faces.
//      connected: There exists a path between any two vertices.
//      orientable: There exists a consistent normal for each face.
// These requirements will be true per construction.
// To implement this mesh, we use the doubly connected edge list (DCEL) data structure, also known as half-edge data structure.
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Douconel<VertID: Key, V, EdgeID: Key, E, FaceID: Key, F> {
    pub verts: SlotMap<VertID, V>,
    pub edges: SlotMap<EdgeID, E>,
    pub faces: SlotMap<FaceID, F>,

    edge_root: SecondaryMap<EdgeID, VertID>,
    edge_face: SecondaryMap<EdgeID, FaceID>,
    edge_next: SecondaryMap<EdgeID, EdgeID>,
    edge_twin: SecondaryMap<EdgeID, EdgeID>,

    vert_rep: SecondaryMap<VertID, EdgeID>,
    face_rep: SecondaryMap<FaceID, EdgeID>,
}

impl<VertID: slotmap::Key, V: Default, EdgeID: Key, E: Default, FaceID: Key, F: Default> Douconel<VertID, V, EdgeID, E, FaceID, F> {
    // Creates a new, empty Douconel.
    #[must_use]
    fn new() -> Self {
        Self {
            verts: SlotMap::with_key(),
            edges: SlotMap::with_key(),
            faces: SlotMap::with_key(),
            edge_root: SecondaryMap::new(),
            edge_face: SecondaryMap::new(),
            edge_next: SecondaryMap::new(),
            edge_twin: SecondaryMap::new(),
            vert_rep: SecondaryMap::new(),
            face_rep: SecondaryMap::new(),
        }
    }

    // Construct a DCEL from a list of faces, where each face is a list of vertex indices.
    pub fn from_faces(faces: &[Vec<usize>]) -> Result<(Self, BiHashMap<usize, VertID>, BiHashMap<usize, FaceID>), MeshError<VertID>> {
        let mut mesh = Self::new();

        // 1. Create the vertices.
        //      trivial; get all unique input vertices (from the faces), and create a vertex for each of them
        //
        // 2. Create the faces with its (half)edges.
        //      each face has edges defined by a sequence of vertices, example:
        //          face = [v0, v1, v2]
        //          then we create three edges = [(v0, v1), (v1, v2), (v2, v0)]
        //                v0
        //                *
        //               ^ \
        //              /   \ e0
        //          e2 /     \
        //            /       v
        //        v2 * < - - - * v1
        //                e1
        //
        //      Also assign representatives to vertices and faces whenever you make them.
        //
        // 3. Assign twins.
        //      trivial; just assign THE edge that has the same endpoints, but swapped (just requires some bookkeeping)
        //      return error if no such edge exists
        //

        // 0. TODO: Make sure the faces are orientable.
        // for each face, choose an orientation, then make sure the neighboring faces have the same orientation.

        // 1. Create the vertices.
        // Need mapping between original indices, and new pointers
        let mut vertex_pointers = BiHashMap::new();
        let mut face_pointers = BiHashMap::new();

        let vertices = faces.iter().flatten().unique().copied().collect_vec();

        for inp_vert_id in vertices {
            let vert_id = mesh.verts.insert(V::default());
            vertex_pointers.insert(inp_vert_id, vert_id);
        }

        // 2. Create the faces with its (half)edges.
        // Need mapping between endpoints and edges for later use (assigning twins).
        let mut endpoints_to_edges = HashMap::<(VertID, VertID), EdgeID>::new();
        for (inp_face_id, inp_face_edges) in faces.iter().enumerate() {
            let face_id = mesh.faces.insert(F::default());

            face_pointers.insert(inp_face_id, face_id);

            let mut conc = inp_face_edges.clone();
            conc.push(inp_face_edges[0]); // Re-append the first to loop back

            let edges = conc
                .iter()
                .tuple_windows()
                .map(|(inp_start_vertex, inp_end_vertex)| {
                    (
                        vertex_pointers
                            .get_by_left(inp_start_vertex)
                            .copied()
                            .unwrap_or_else(|| panic!("V:{inp_start_vertex} does not have a vertex pointer")),
                        vertex_pointers
                            .get_by_left(inp_end_vertex)
                            .copied()
                            .unwrap_or_else(|| panic!("V:{inp_end_vertex} does not have a vertex pointer")),
                    )
                })
                .collect_vec();

            let mut edge_ids = vec![];
            for (start_vertex, end_vertex) in edges {
                let edge_id = mesh.edges.insert(E::default());

                if endpoints_to_edges.insert((start_vertex, end_vertex), edge_id).is_some() {
                    return Err(MeshError::DuplicateEdge(start_vertex, end_vertex));
                };

                mesh.face_rep.insert(face_id, edge_id);
                mesh.vert_rep.insert(start_vertex, edge_id);
                mesh.edge_root.insert(edge_id, start_vertex);
                mesh.edge_face.insert(edge_id, face_id);
                edge_ids.push(edge_id);
            }

            // Linking each edge to its next edge in the face
            for edge_index in 0..edge_ids.len() {
                mesh.edge_next.insert(edge_ids[edge_index], edge_ids[(edge_index + 1) % edge_ids.len()]);
            }
        }

        // 3. Assign twins.
        for (&(vert_a, vert_b), &edge_id) in &endpoints_to_edges {
            // Retrieve the twin edge
            if let Some(twin_id) = endpoints_to_edges.get(&(vert_b, vert_a)).copied() {
                // Assign twins
                mesh.edge_twin.insert(edge_id, twin_id);
                mesh.edge_twin.insert(twin_id, edge_id);
            } else {
                return Err(MeshError::NoTwin(vert_a, vert_b));
            }
        }

        // 4. Make sure the mesh is connected.
        if !mesh.is_connected() {
            return Err(MeshError::NotConnected);
        }

        // 5. Make sure the mesh is orientable.
        // TODO:

        // Assert that all elements have their required properties set.
        mesh.assert_properties();
        mesh.assert_references();
        mesh.assert_invariants();

        Ok((mesh, vertex_pointers, face_pointers))
    }

    // Asserts that all elements have their required properties set.
    // These assertions should all pass per construction.
    pub fn assert_properties(&self) {
        for edge_id in self.edge_ids() {
            assert!(self.edge_root.contains_key(edge_id), "{edge_id:?} has no root");
            assert!(self.edge_face.contains_key(edge_id), "{edge_id:?} has no face");
            assert!(self.edge_next.contains_key(edge_id), "{edge_id:?} has no next");
            assert!(self.edge_twin.contains_key(edge_id), "{edge_id:?} has no twin");
        }
        for vert_id in self.vert_ids() {
            assert!(self.vert_rep.contains_key(vert_id), "{vert_id:?} has no vrep");
        }
        for face_id in self.face_ids() {
            assert!(self.face_rep.contains_key(face_id), "{face_id:?} has no frep");
        }
    }

    // Asserts that all references between elements are valid.
    // These assertions should all pass per construction.
    pub fn assert_references(&self) {
        for edge_id in self.edge_ids() {
            let root_id = self.root(edge_id);
            assert!(self.verts.contains_key(root_id), "{edge_id:?} has non-existing root ({root_id:?})");

            let face_id = self.face(edge_id);
            assert!(self.faces.contains_key(face_id), "{edge_id:?} has non-existing face ({face_id:?})");

            let next_id = self.next(edge_id);
            assert!(self.edges.contains_key(next_id), "{edge_id:?} has non-existing next ({next_id:?})");

            let twin_id = self.twin(edge_id);
            assert!(self.edges.contains_key(twin_id), "{edge_id:?} has non-existing twin ({twin_id:?})");
        }
        for vert_id in self.vert_ids() {
            let rep_id = self.vert_rep[vert_id];
            assert!(self.edges.contains_key(rep_id), "{vert_id:?} has non-existing vrep ({rep_id:?})");
        }
        for face_id in self.face_ids() {
            let rep_id = self.face_rep[face_id];
            assert!(self.edges.contains_key(rep_id), "{face_id:?} has non-existing frep ({rep_id:?})");
        }
    }

    // Asserts the invariants of the DCEL structure.
    pub fn assert_invariants(&self) {
        // this->twin->twin == this
        for edge_id in self.edge_ids() {
            assert!(self.twin(self.twin(edge_id)) == edge_id, "{edge_id:?}: [this->twin->twin == this] violated");
        }
        // this->twin->next->root == this->root
        for edge_id in self.edge_ids() {
            assert!(
                self.root(self.next(self.twin(edge_id))) == self.root(edge_id),
                "{edge_id:?}: [this->twin->next->root == this->root] violated"
            );
        }
        // this->next->face == this->face
        for edge_id in self.edge_ids() {
            assert!(
                self.face(self.next(edge_id)) == self.face(edge_id),
                "{edge_id:?}: [this->next->face == this->face] violated"
            );
        }
        // this->next->...->next == this
        const MAX_FACE_SIZE: usize = 10;
        for edge_id in self.edge_ids() {
            let mut next_id = edge_id;
            for _ in 0..MAX_FACE_SIZE {
                next_id = self.next(next_id);
                if next_id == edge_id {
                    break;
                }
            }
            assert!(next_id == edge_id, "{edge_id:?}: [this->next->...->next == this] violated");
        }
    }

    // Returns the "representative" edge of the given vertex.
    // Panics if the vertex has no representative edge defined.
    #[must_use]
    pub fn vrep(&self, id: VertID) -> EdgeID {
        self.vert_rep.get(id).copied().unwrap_or_else(|| panic!("{id:?} has no vrep"))
    }

    // Returns the "representative" edge of the given face.
    // Panics if the face has no representative edge defined.
    #[must_use]
    pub fn frep(&self, id: FaceID) -> EdgeID {
        self.face_rep.get(id).copied().unwrap_or_else(|| panic!("{id:?} has no frep"))
    }

    // Returns the root vertex of the given edge.
    // Panics if the edge has no root defined or if the root does not exist.
    #[must_use]
    pub fn root(&self, id: EdgeID) -> VertID {
        self.edge_root.get(id).copied().unwrap_or_else(|| panic!("{id:?} has no root"))
    }

    // Returns the root of the twin edge of the given edge. (also named toor, reverse of root)
    // Panics if the edge has no twin defined or if the twin does not exist.
    #[must_use]
    pub fn toor(&self, id: EdgeID) -> VertID {
        self.root(self.twin(id))
    }

    // Returns the twin edge of the given edge.
    // Panics if the edge has no twin defined or if the twin does not exist.
    #[must_use]
    pub fn twin(&self, id: EdgeID) -> EdgeID {
        self.edge_twin.get(id).copied().unwrap_or_else(|| panic!("{id:?} has no twin"))
    }

    // Returns the next edge of the given edge.
    // Panics if the edge has no next defined or if the next does not exist.
    #[inline]
    #[must_use]
    pub fn next(&self, id: EdgeID) -> EdgeID {
        self.edge_next.get(id).copied().unwrap_or_else(|| panic!("{id:?} has no next"))
    }

    #[inline]
    #[must_use]
    pub fn nexts(&self, id: EdgeID) -> Vec<EdgeID> {
        let mut nexts = vec![];
        let mut cur = id;
        loop {
            cur = self.next(cur);
            if cur == id {
                return nexts;
            }
            nexts.push(cur);
        }
    }

    // Returns the four edges around a given edge.
    #[inline]
    #[must_use]
    pub fn quad(&self, id: EdgeID) -> [EdgeID; 4] {
        let edge0 = self.next(id);
        let edge1 = self.next(edge0);
        let twin = self.twin(id);
        let edge2 = self.next(twin);
        let edge3 = self.next(edge2);
        [edge0, edge1, edge2, edge3]
    }

    // Returns the face of the given edge.
    // Panics if the edge has no face defined or if the face does not exist.
    #[inline]
    #[must_use]
    pub fn face(&self, id: EdgeID) -> FaceID {
        self.edge_face.get(id).copied().unwrap_or_else(|| panic!("{id:?} has no face"))
    }

    // Returns the start and end vertex IDs of the given edge.
    // Panics if any of the roots are not defined or do not exist.
    #[inline]
    #[must_use]
    pub fn endpoints(&self, id: EdgeID) -> (VertID, VertID) {
        (self.root(id), self.root(self.twin(id)))
    }

    // Returns the corner vertices of a given face.
    #[inline]
    #[must_use]
    pub fn corners(&self, id: FaceID) -> Vec<VertID> {
        self.edges(id).into_iter().map(|edge_id| self.root(edge_id)).collect()
    }

    // Returns the outgoing edges of a given vertex. (clockwise order)
    #[inline]
    #[must_use]
    pub fn outgoing(&self, id: VertID) -> Vec<EdgeID> {
        let mut edges = vec![self.vrep(id)];
        loop {
            let next_of_twin = self.next(self.twin(edges.last().copied().unwrap()));
            if edges.contains(&next_of_twin) {
                return edges;
            }
            edges.push(next_of_twin);
        }
    }

    // Returns the edges of a given face. (anticlockwise order)
    #[inline]
    #[must_use]
    pub fn edges(&self, id: FaceID) -> Vec<EdgeID> {
        [vec![self.frep(id)], self.nexts(self.frep(id))].concat()
    }

    // Returns the faces around a given vertex. (clockwise order)
    #[inline]
    #[must_use]
    pub fn star(&self, id: VertID) -> Vec<FaceID> {
        self.outgoing(id).iter().map(|&edge_id| self.face(edge_id)).collect()
    }

    // Returns the faces around a given edge.
    #[inline]
    #[must_use]
    pub fn faces(&self, id: EdgeID) -> [FaceID; 2] {
        [self.face(id), self.face(self.twin(id))]
    }

    // Returns the face with given vertices.
    #[inline]
    #[must_use]
    pub fn face_with_verts(&self, verts: &[VertID]) -> Option<FaceID> {
        self.star(verts[0])
            .into_iter()
            .find(|&face_id| verts.iter().all(|&vert_id| self.star(vert_id).contains(&face_id)))
    }

    // Returns the two edges of a given face that are connected to the given vertex.
    #[inline]
    #[must_use]
    pub fn edges_in_face_with_vert(&self, face_id: FaceID, vert_id: VertID) -> Option<[EdgeID; 2]> {
        let edges = self.edges(face_id);
        edges
            .into_iter()
            .filter(|&edge_id| self.root(edge_id) == vert_id || self.toor(edge_id) == vert_id)
            .collect_tuple()
            .map(|(a, b)| [a, b])
    }

    #[inline]
    #[must_use]
    pub fn common_endpoint(&self, edge_a: EdgeID, edge_b: EdgeID) -> Option<VertID> {
        let (a0, a1) = self.endpoints(edge_a);
        let (b0, b1) = self.endpoints(edge_b);
        if a0 == b0 || a0 == b1 {
            Some(a0)
        } else if a1 == b0 || a1 == b1 {
            Some(a1)
        } else {
            None
        }
    }

    #[inline]
    #[must_use]
    pub fn verts_to_edges(&self, verts: &[VertID]) -> Vec<EdgeID> {
        verts
            .iter()
            .flat_map(|&vert_id| {
                self.outgoing(vert_id)
                    .into_iter()
                    .filter(|&edge_id| verts.contains(&self.toor(edge_id)))
                    .collect_vec()
            })
            .collect_vec()
    }

    // Returns the edge between the two vertices. Returns None if the vertices are not connected.
    #[inline]
    #[must_use]
    pub fn edge_between_verts(&self, id_a: VertID, id_b: VertID) -> Option<(EdgeID, EdgeID)> {
        for &edge_a_id in &self.outgoing(id_a) {
            for &edge_b_id in &self.outgoing(id_b) {
                if self.twin(edge_a_id) == edge_b_id {
                    return Some((edge_a_id, edge_b_id));
                }
            }
        }
        None
    }

    // Returns the edge between the two faces. Returns None if the faces do not share an edge.
    #[must_use]
    pub fn edge_between_faces(&self, id_a: FaceID, id_b: FaceID) -> Option<(EdgeID, EdgeID)> {
        let edges_a = self.edges(id_a);
        let edges_b = self.edges(id_b);
        for &edge_a_id in &edges_a {
            for &edge_b_id in &edges_b {
                if self.twin(edge_a_id) == edge_b_id {
                    return Some((edge_a_id, edge_b_id));
                }
            }
        }
        None
    }

    // Returns the neighbors of a given vertex.
    #[must_use]
    pub fn vneighbors(&self, id: VertID) -> Vec<VertID> {
        self.outgoing(id).iter().map(|&edge_id| self.root(self.twin(edge_id))).collect()
    }

    // Returns the (edge-wise) neighbors of a given face.
    #[must_use]
    pub fn fneighbors(&self, id: FaceID) -> Vec<FaceID> {
        self.edges(id).into_iter().map(|edge_id| self.face(self.twin(edge_id))).collect()
    }

    // Returns the number of vertices in the mesh.
    #[must_use]
    pub fn nr_verts(&self) -> usize {
        self.verts.len()
    }

    // Returns the number of (half)edges in the mesh.
    #[must_use]
    pub fn nr_edges(&self) -> usize {
        self.edges.len()
    }

    // Returns the number of faces in the mesh.
    #[must_use]
    pub fn nr_faces(&self) -> usize {
        self.faces.len()
    }

    #[must_use]
    pub fn vert_ids(&self) -> Vec<VertID> {
        self.verts.keys().collect()
    }

    #[must_use]
    pub fn edge_ids(&self) -> Vec<EdgeID> {
        self.edges.keys().collect()
    }

    #[must_use]
    pub fn face_ids(&self) -> Vec<FaceID> {
        self.faces.keys().collect()
    }

    // Return `n` random vertices.
    #[must_use]
    pub fn random_verts(&self, n: usize) -> Vec<VertID> {
        self.verts.keys().choose_multiple(&mut rand::thread_rng(), n)
    }

    // Return `n` random edges.
    #[must_use]
    pub fn random_edges(&self, n: usize) -> Vec<EdgeID> {
        self.edges.keys().choose_multiple(&mut rand::thread_rng(), n)
    }

    // Return `n` random faces.
    #[must_use]
    pub fn random_faces(&self, n: usize) -> Vec<FaceID> {
        self.faces.keys().choose_multiple(&mut rand::thread_rng(), n)
    }

    pub fn neighbor_function_primal(&self) -> impl Fn(VertID) -> Vec<VertID> + '_ {
        |v_id| self.vneighbors(v_id)
    }

    pub fn neighbor_function_edgegraph(&self) -> impl Fn(EdgeID) -> Vec<EdgeID> + '_ {
        |e_id| vec![self.next(e_id), self.next(self.next(e_id)), self.twin(e_id)]
    }

    #[inline]
    pub fn neighbor_function_edgepairgraph(&self) -> impl Fn([EdgeID; 2]) -> Vec<[EdgeID; 2]> + '_ {
        |[_, to]| {
            let next = self.twin(to);
            self.nexts(next).into_iter().map(|next_to| [next, next_to]).collect()
        }
    }

    #[must_use]
    pub fn is_connected(&self) -> bool {
        hutspot::graph::find_ccs(&self.vert_ids(), self.neighbor_function_primal()).len() == 1
    }

    #[must_use]
    pub fn wedges(&self, a: VertID, b: VertID, c: VertID) -> (Vec<VertID>, Vec<VertID>) {
        // First wedge is a to c (around b)
        let wedge1 = std::iter::once(a)
            .chain(self.vneighbors(b).into_iter().cycle().skip_while(|&v| v != a).skip(1).take_while(|&v| v != c))
            .chain([c])
            .collect_vec();

        // Second wedge is c to a (around b)
        let wedge2 = std::iter::once(c)
            .chain(self.vneighbors(b).into_iter().cycle().skip_while(|&v| v != c).skip(1).take_while(|&v| v != a))
            .chain([a])
            .collect_vec();

        // Return the wedges
        (wedge1, wedge2)
    }
}

impl<VertID: Key, V: Default, EdgeID: Key, E: Default, FaceID: Key, F: Default + Clone> Douconel<VertID, V, EdgeID, E, FaceID, F> {
    pub fn split_edge(&mut self, edge_id: EdgeID) -> (VertID, [FaceID; 4]) {
        // First face
        let e_ab = edge_id;
        let e_b0 = self.next(e_ab);
        let e_0a = self.next(e_b0);
        assert!(self.next(e_0a) == e_ab);

        let v_a = self.root(e_ab);
        let v_b = self.root(e_b0);
        let v_0 = self.root(e_0a);

        // Second face
        let e_ba = self.twin(edge_id);
        let e_a1 = self.next(e_ba);
        let e_1b = self.next(e_a1);
        assert!(self.next(e_1b) == e_ba);

        assert!(self.root(e_ba) == v_b);
        assert!(self.root(e_a1) == v_a);
        let v_1 = self.root(e_1b);

        // Four new faces (re-use original id for first 2)
        let f_0 = self.face(e_ab);
        self.face_rep.insert(f_0, e_0a);

        let f_1 = self.face(e_ba);
        self.face_rep.insert(f_1, e_a1);

        let f_2 = self.faces.insert(F::default());
        self.face_rep.insert(f_2, e_b0);

        let f_3 = self.faces.insert(F::default());
        self.face_rep.insert(f_3, e_1b);

        // Six new edges (with next six available ids)

        // f_0
        let e_ax = e_ab;
        let e_x0 = self.edges.insert(E::default());

        // f_1
        let e_xa = e_ba;
        let e_1x = self.edges.insert(E::default());

        // f_2
        let e_xb = self.edges.insert(E::default());
        let e_0x = self.edges.insert(E::default());

        // f_3
        let e_bx = self.edges.insert(E::default());
        let e_x1 = self.edges.insert(E::default());

        // One new vertex (with next available id)
        let v_x = self.verts.insert(V::default());
        self.vert_rep.insert(v_x, e_xa);

        self.vert_rep.insert(v_b, e_b0);
        self.vert_rep.insert(v_a, e_a1);

        // Set the edges correctly
        self.edge_root.insert(e_ax, v_a);
        self.edge_face.insert(e_ax, f_0);
        self.edge_next.insert(e_ax, e_x0);
        self.edge_twin.insert(e_ax, e_xa);

        self.edge_root.insert(e_xa, v_x);
        self.edge_face.insert(e_xa, f_1);
        self.edge_next.insert(e_xa, e_a1);
        self.edge_twin.insert(e_xa, e_ax);

        self.edge_root.insert(e_bx, v_b);
        self.edge_face.insert(e_bx, f_3);
        self.edge_next.insert(e_bx, e_x1);
        self.edge_twin.insert(e_bx, e_xb);

        self.edge_root.insert(e_xb, v_x);
        self.edge_face.insert(e_xb, f_2);
        self.edge_next.insert(e_xb, e_b0);
        self.edge_twin.insert(e_xb, e_bx);

        self.edge_root.insert(e_0x, v_0);
        self.edge_face.insert(e_0x, f_2);
        self.edge_next.insert(e_0x, e_xb);
        self.edge_twin.insert(e_0x, e_x0);

        self.edge_root.insert(e_x0, v_x);
        self.edge_face.insert(e_x0, f_0);
        self.edge_next.insert(e_x0, e_0a);
        self.edge_twin.insert(e_x0, e_0x);

        self.edge_root.insert(e_1x, v_1);
        self.edge_face.insert(e_1x, f_1);
        self.edge_next.insert(e_1x, e_xa);
        self.edge_twin.insert(e_1x, e_x1);

        self.edge_root.insert(e_x1, v_x);
        self.edge_face.insert(e_x1, f_3);
        self.edge_next.insert(e_x1, e_1b);
        self.edge_twin.insert(e_x1, e_1x);

        self.edge_face.insert(e_a1, f_1);
        self.edge_next.insert(e_a1, e_1x);

        self.edge_face.insert(e_1b, f_3);
        self.edge_next.insert(e_1b, e_bx);

        self.edge_face.insert(e_b0, f_2);
        self.edge_next.insert(e_b0, e_0x);

        self.edge_face.insert(e_0a, f_0);
        self.edge_next.insert(e_0a, e_ax);

        (v_x, [f_0, f_1, f_2, f_3])
    }

    pub fn split_face(&mut self, face_id: FaceID) -> (VertID, [FaceID; 3]) {
        let edges = self.edges(face_id);
        // let centroid = self.centroid(face_id);

        // Original face
        let e_01 = edges[0];
        let v_0 = self.root(e_01);

        let e_12 = edges[1];
        let v_1 = self.root(e_12);

        let e_20 = edges[2];
        let v_2 = self.root(e_20);

        // Two new faces (original face stays the same)
        let f_0 = face_id;

        let f_1 = self.faces.insert(self.faces[face_id].clone());
        let f_2 = self.faces.insert(self.faces[face_id].clone());

        self.face_rep.insert(f_1, e_12);
        self.face_rep.insert(f_2, e_20);

        // Six new edges (with next six available ids)
        let e_x0 = self.edges.insert(E::default());
        let e_x1 = self.edges.insert(E::default());
        let e_x2 = self.edges.insert(E::default());
        let e_0x = self.edges.insert(E::default());
        let e_1x = self.edges.insert(E::default());
        let e_2x = self.edges.insert(E::default());

        let v_x = self.verts.insert(V::default());
        self.vert_rep.insert(v_x, e_x0);

        self.edge_root.insert(e_x0, v_x);
        self.edge_face.insert(e_x0, f_0);
        self.edge_next.insert(e_x0, e_01);
        self.edge_twin.insert(e_x0, e_0x);

        self.edge_root.insert(e_x1, v_x);
        self.edge_face.insert(e_x1, f_1);
        self.edge_next.insert(e_x1, e_12);
        self.edge_twin.insert(e_x1, e_1x);

        self.edge_root.insert(e_x2, v_x);
        self.edge_face.insert(e_x2, f_2);
        self.edge_next.insert(e_x2, e_20);
        self.edge_twin.insert(e_x2, e_2x);

        self.edge_root.insert(e_0x, v_0);
        self.edge_face.insert(e_0x, f_2);
        self.edge_next.insert(e_0x, e_x2);
        self.edge_twin.insert(e_0x, e_x0);

        self.edge_root.insert(e_1x, v_1);
        self.edge_face.insert(e_1x, f_0);
        self.edge_next.insert(e_1x, e_x0);
        self.edge_twin.insert(e_1x, e_x1);

        self.edge_root.insert(e_2x, v_2);
        self.edge_face.insert(e_2x, f_1);
        self.edge_next.insert(e_2x, e_x1);
        self.edge_twin.insert(e_2x, e_x2);

        self.edge_face.insert(e_01, f_0);
        self.edge_face.insert(e_12, f_1);
        self.edge_face.insert(e_20, f_2);
        self.edge_next.insert(e_01, e_1x);
        self.edge_next.insert(e_12, e_2x);
        self.edge_next.insert(e_20, e_0x);

        (v_x, [f_0, f_1, f_2])
    }
}
