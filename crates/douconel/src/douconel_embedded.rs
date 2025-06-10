use crate::douconel::{Douconel, MeshError};
use bimap::BiHashMap;
use hutspot::geom::Vector2D;
use itertools::Itertools;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use slotmap::Key;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbeddedMeshError<VertID, FaceID> {
    #[error("{0} is not a polygon (less than 3 vertices)")]
    FaceNotPolygon(FaceID),
    #[error("{0} is not planar (vertices are not coplanar)")]
    FaceNotPlanar(FaceID),
    #[error("{0} is not simple (edges intersect)")]
    FaceNotSimple(FaceID),
    #[error("{0:?}")]
    MeshError(MeshError<VertID>),
}

type Float = f64;
type Vector3D = nalgebra::SVector<Float, 3>;
const PI: f64 = std::f64::consts::PI;

pub trait HasPosition {
    fn position(&self) -> Vector3D;
    fn set_position(&mut self, position: Vector3D);
}

// Embedded vertices (have a position)
#[derive(Default, Copy, Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddedVertex {
    position: Vector3D,
}

impl HasPosition for EmbeddedVertex {
    fn position(&self) -> Vector3D {
        self.position
    }
    fn set_position(&mut self, position: Vector3D) {
        self.position = position;
    }
}

impl<VertID: Key, V: Default + HasPosition, EdgeID: Key, E: Default, FaceID: Key, F: Default> Douconel<VertID, V, EdgeID, E, FaceID, F> {
    // This is a struct that defines an embedded mesh with vertices (with position), edges, and faces (with clockwise ordering).
    // This embedded mesh is:
    //      a closed 2-manifold: Each edge corresponds to exactly two faces.
    //      connected: There exists a path between any two vertices.
    //      orientable: There exists a consistent normal for each face.
    //      polygonal: Each face is a simple polygon (lies in a plane, no intersections).
    // These requirements will be true per construction.
    pub fn from_embedded_faces(
        faces: &[Vec<usize>],
        vertex_positions: &[Vector3D],
    ) -> Result<(Self, BiHashMap<usize, VertID>, BiHashMap<usize, FaceID>), EmbeddedMeshError<VertID, FaceID>> {
        let non_embedded = Self::from_faces(faces);
        if let Ok((mut douconel, vertex_map, face_map)) = non_embedded {
            for (inp_vertex_id, inp_vertex_position) in vertex_positions.iter().copied().enumerate() {
                let vertex_id = vertex_map.get_by_left(&inp_vertex_id).copied().unwrap_or_default();
                if let Some(vert) = douconel.verts.get_mut(vertex_id) {
                    vert.set_position(inp_vertex_position);
                }
            }

            // Make sure the mesh is polygonal
            for face_id in douconel.faces.keys() {
                let corners = douconel.corners(face_id);

                // Check that the face is a polygon
                if corners.len() < 3 {
                    return Err(EmbeddedMeshError::FaceNotPolygon(face_id));
                }

                // Check that the face is planar
                let a = corners[0];
                let b = corners[1];
                let c = corners[2];
                for d in corners.into_iter().skip(3) {
                    if !hutspot::geom::are_points_coplanar(douconel.position(a), douconel.position(b), douconel.position(c), douconel.position(d)) {
                        return Err(EmbeddedMeshError::FaceNotSimple(face_id));
                    }
                }

                // Check that the face is simple
                for edge_a in douconel.edges(face_id) {
                    for edge_b in douconel.edges(face_id) {
                        if edge_a == edge_b {
                            continue;
                        }
                        let a_u = douconel.position(douconel.root(edge_a));
                        let a_v = douconel.position(douconel.toor(edge_a));
                        let b_u = douconel.position(douconel.root(edge_b));
                        let b_v = douconel.position(douconel.toor(edge_b));
                        if let Some((_, hutspot::geom::IntersectionType::Proper)) = hutspot::geom::calculate_3d_lineseg_intersection(a_u, a_v, b_u, b_v) {
                            return Err(EmbeddedMeshError::FaceNotSimple(face_id));
                        }
                    }
                }
            }

            Ok((douconel, vertex_map, face_map))
        } else {
            non_embedded.map_err(EmbeddedMeshError::MeshError)
        }
    }

    // Get position of a given vertex.
    #[must_use]
    pub fn position(&self, id: VertID) -> Vector3D {
        self.verts.get(id).unwrap_or_else(|| panic!("V:{id:?} not initialized")).position()
    }

    // Get centroid of a given polygonal face.
    // https://en.wikipedia.org/wiki/Centroid
    // Be careful with concave faces, the centroid might lay outside the face.
    #[must_use]
    pub fn centroid(&self, face_id: FaceID) -> Vector3D {
        hutspot::math::calculate_average_f64(self.edges(face_id).iter().map(|&edge_id| self.position(self.root(edge_id))))
    }

    // Get midpoint of a given edge.
    #[must_use]
    pub fn midpoint(&self, edge_id: EdgeID) -> Vector3D {
        self.midpoint_offset(edge_id, 0.5)
    }

    // Get midpoint of a given edge with some offset
    #[must_use]
    pub fn midpoint_offset<T>(&self, edge_id: EdgeID, offset: T) -> Vector3D
    where
        T: Into<Float>,
    {
        self.position(self.root(edge_id)) + self.vector(edge_id) * offset.into()
    }

    // Get vector of a given edge.
    #[must_use]
    pub fn vector(&self, id: EdgeID) -> Vector3D {
        let (u, v) = self.endpoints(id);
        self.position(v) - self.position(u)
    }

    // Get length of a given edge.
    #[must_use]
    pub fn length(&self, id: EdgeID) -> Float {
        self.vector(id).magnitude()
    }

    // Get distance between two vertices.
    #[must_use]
    pub fn distance(&self, v_a: VertID, v_b: VertID) -> Float {
        self.position(v_a).metric_distance(&self.position(v_b))
    }

    // Get angle (in radians) between two vectors `a` and `b`.
    #[must_use]
    pub fn vec_angle(&self, a: Vector3D, b: Vector3D) -> Float {
        a.angle(&b)
    }

    // Get angle (in radians) between two edges `u` and `v`.
    #[must_use]
    pub fn angle(&self, u: EdgeID, v: EdgeID) -> Float {
        self.vec_angle(self.vector(u), self.vector(v))
    }

    // Get angular defect of a vertex (2PI - C, where C = the sum of all the angles at the vertex).
    // See https://en.wikipedia.org/wiki/Angular_defect
    #[must_use]
    pub fn defect(&self, id: VertID) -> Float {
        let sum_of_angles = self.outgoing(id).iter().fold(0., |sum, &outgoing_edge_id| {
            let incoming_edge_id = self.twin(outgoing_edge_id);
            let next_edge_id = self.next(incoming_edge_id);
            let angle = self.angle(outgoing_edge_id, next_edge_id);
            sum + angle
        });

        // 2PI - C
        Float::from(2.0).mul_add(PI, -sum_of_angles)
    }

    // Vector area of a given face.
    #[must_use]
    pub fn vector_area(&self, id: FaceID) -> Vector3D {
        self.edges(id).iter().fold(Vector3D::zeros(), |sum, &edge_id| {
            let u = self.vector(self.twin(edge_id));
            let v = self.vector(self.next(edge_id));
            sum + u.cross(&v)
        })
    }

    // Area of a given face.
    #[must_use]
    pub fn area(&self, id: FaceID) -> Float {
        self.vector_area(id).magnitude() / 2.0
    }

    // Get normal of face `id`. Assumes the face is planar. If the face is not planar, then this function will not return the correct normal.
    // The normal is calculated as the normalized vector area of the face; https://en.wikipedia.org/wiki/Normal_(geometry)
    #[must_use]
    pub fn normal(&self, id: FaceID) -> Vector3D {
        -self.vector_area(id).normalize()
    }

    // Get the average normals around vertex `id`.
    #[must_use]
    pub fn vert_normal(&self, id: VertID) -> Vector3D {
        self.star(id).iter().map(|&face_id| self.normal(face_id)).sum::<Vector3D>().normalize()
    }

    // Get the normal of edge `id` by averaging the normals of the two faces it belongs to.
    #[must_use]
    pub fn edge_normal(&self, id: EdgeID) -> Vector3D {
        let [f1, f2] = self.faces(id);
        (self.normal(f1) + self.normal(f2)).normalize()
    }

    // Get the angle between two edges at a common vertex.
    #[must_use]
    pub fn vertex_angle(&self, a: VertID, b: VertID, c: VertID) -> Float {
        let a_b = self.position(b) - self.position(a);
        let b_c = self.position(b) - self.position(c);
        self.vec_angle(a_b, b_c)
    }

    // Weight function
    pub fn weight_function_euclidean(&self) -> impl Fn(VertID, VertID) -> OrderedFloat<Float> + '_ {
        |a, b| OrderedFloat(self.distance(a, b))
    }

    // Weight function
    pub fn weight_function_angle_edges(&self, slack: i32) -> impl Fn(EdgeID, EdgeID) -> OrderedFloat<Float> + '_ {
        move |a, b| OrderedFloat(self.angle(a, b).powi(slack))
    }

    // Weight function
    pub fn weight_function_angle_edgepairs(&self, slack: i32) -> impl Fn((EdgeID, EdgeID), (EdgeID, EdgeID)) -> OrderedFloat<Float> + '_ {
        move |a, b| {
            let vector_a = self.midpoint(a.1) - self.midpoint(a.0);
            let vector_b = self.midpoint(b.1) - self.midpoint(b.0);
            OrderedFloat(self.vec_angle(vector_a, vector_b).powi(slack))
        }
    }

    // Weight function
    pub fn weight_function_angle_edgepairs_aligned(
        &self,
        angular_slack: i32,
        alignment_slack: i32,
        axis: Vector3D,
    ) -> impl Fn([EdgeID; 2], [EdgeID; 2]) -> OrderedFloat<Float> + '_ {
        move |a, b| {
            let vector_a = self.midpoint(a[1]) - self.midpoint(a[0]);
            let vector_b = self.midpoint(b[1]) - self.midpoint(b[0]);

            let weight = self.vec_angle(vector_a, vector_b).powi(angular_slack)
                + (self.vec_angle(vector_a.cross(&self.edge_normal(a[0])), axis)).powi(alignment_slack)
                + (self.vec_angle(vector_b.cross(&self.edge_normal(b[0])), axis)).powi(alignment_slack);

            OrderedFloat(weight)
        }
    }

    // Weight function
    pub fn weight_function_angle_edgepairs_aligned_components(
        &self,
        axis: Vector3D,
    ) -> impl Fn([EdgeID; 2], [EdgeID; 2]) -> (OrderedFloat<Float>, OrderedFloat<Float>, OrderedFloat<Float>) + '_ {
        move |a, b| {
            let vector_a = self.midpoint(a[1]) - self.midpoint(a[0]);
            let vector_b = self.midpoint(b[1]) - self.midpoint(b[0]);

            (
                OrderedFloat(self.vec_angle(vector_a, vector_b)),
                OrderedFloat(self.vec_angle(vector_a.cross(&self.edge_normal(a[0])), axis)),
                OrderedFloat(self.vec_angle(vector_b.cross(&self.edge_normal(b[0])), axis)),
            )
        }
    }

    #[must_use]
    pub fn wedge_alpha(&self, (b, wedge): (VertID, &[VertID])) -> f64 {
        wedge.windows(2).map(|vs| self.vertex_angle(vs[0], b, vs[1])).sum::<f64>()
    }

    #[must_use]
    pub fn shortest_wedge(&self, a: VertID, b: VertID, c: VertID) -> (Vec<VertID>, f64) {
        let (w1, w2) = self.wedges(a, b, c);
        let (a1, a2) = (self.wedge_alpha((b, &w1)), self.wedge_alpha((b, &w2)));
        if a1 < a2 {
            (w1, a1)
        } else {
            (w2.into_iter().rev().collect_vec(), a2)
        }
    }
}

impl<VertID: Key, V: Default + HasPosition, EdgeID: Key, E: Default, FaceID: Key, F: Default + Clone> Douconel<VertID, V, EdgeID, E, FaceID, F> {
    pub fn splip_edge(&mut self, a: VertID, b: VertID) -> Option<VertID> {
        // Make sure the edge exists
        let edge = self.edge_between_verts(a, b).unwrap().0;

        // Get the two faces adjacent to the two edges
        let [f1, f2] = self.faces(edge);

        // Get the anchor vertex of f1 (the vertex that is not a or b)
        let c1 = self.corners(f1).iter().find(|&&v| v != a && v != b).unwrap().to_owned();
        // Get the anchor vertex of f2 (the vertex that is not a or b)
        let c2 = self.corners(f2).iter().find(|&&v| v != a && v != b).unwrap().to_owned();

        // Get all required edges
        let a_c1 = self.edge_between_verts(a, c1).unwrap().0;
        let b_c1 = self.edge_between_verts(b, c1).unwrap().0;
        let a_c2 = self.edge_between_verts(a, c2).unwrap().0;
        let b_c2 = self.edge_between_verts(b, c2).unwrap().0;
        let a_b = edge;

        // Construct planar embedding respecting all edge lengths
        let a_c1_distance = self.length(a_c1);
        let b_c1_distance = self.length(b_c1);
        let a_c2_distance = self.length(a_c2);
        let b_c2_distance = self.length(b_c2);
        let a_b_distance = self.length(a_b);

        // if a_c1_distance < 1e-6 || b_c1_distance < 1e-6 || a_c2_distance < 1e-6 || b_c2_distance < 1e-6 || a_b_distance < 1e-6 {
        //     println!("oopsie ");
        //     return None;
        // }

        let a_position = Vector2D::new(0., 0.);
        let b_position = Vector2D::new(a_b_distance, 0.);

        // Calculate the position of c1 (under a_b)
        // Draw circle with radius a_c1_distance and center a_position
        // Draw circle with radius b_c1_distance and center b_position
        // Find intersection point with negative y: this is the position of c1
        let R = a_c1_distance;
        let r = b_c1_distance;
        let d = a_b_distance;

        let x = (d * d - r * r + R * R) / (2. * d);
        let yy = R * R - x * x;
        let y = if yy < 0. { 0. } else { -(yy.sqrt()) };

        let c1_position = Vector2D::new(x, y);
        assert!(c1_position[1] <= 0., "c1_position: {:?}", c1_position);

        // Calculate the position of c2
        // Draw circle with radius a_c2_distance and center a_position
        // Draw circle with radius b_c2_distance and center b_position
        // Find intersection point with positive y: this is the position of c2
        let R = a_c2_distance;
        let r = b_c2_distance;
        let d = a_b_distance;

        let x = (d * d - r * r + R * R) / (2. * d);
        let yy = R * R - x * x;
        let y = if yy < 0. { 0. } else { yy.sqrt() };
        let c2_position = Vector2D::new(x, y);
        assert!(c2_position[1] >= 0., "c2_position: {:?}", c2_position);

        // println!("a_position: {a_position:?}");
        // println!("b_position: {b_position:?}");
        // println!("c1_position: {c1_position:?}");
        // println!("c2_position: {c2_position:?}");

        // Find intersection of a_b and c1_c2
        // Calculate the intersection of the lines a_b and c1_c2

        let intersection_maybe = hutspot::geom::calculate_2d_lineseg_intersection(a_position, b_position, c1_position, c2_position);

        if intersection_maybe.is_none() {
            return None;
        }

        let intersection = intersection_maybe.unwrap().0;

        // assert!(intersection[1].abs() == 0., "{intersection:?}");

        // The portion of the edge a_b that is before the intersection
        let t = intersection[0] / a_b_distance;

        // println!("t: {}", t);

        if t < 0.001 {
            return Some(a);
        }

        if t > 0.999 {
            return Some(b);
        }

        // Calculate the position of the split vertex in 3D
        let split_position = self.position(a) + (self.position(b) - self.position(a)) * t;

        // Split edge a_b
        let (split_vertex, _) = self.split_edge(a_b);

        // There exists an edge between c1 and split_vertex and c2 and split_vertex
        assert!(self.edge_between_verts(c1, split_vertex).is_some());
        assert!(self.edge_between_verts(c2, split_vertex).is_some());

        // Move the split vertex to the correct position
        self.verts.get_mut(split_vertex).unwrap().set_position(split_position);

        return Some(split_vertex);
    }

    pub fn refine(&mut self, n: usize) {
        for _ in 0..n {
            // find the longest edge
            let longest_edge = self.edges.keys().max_by_key(|&edge_id| OrderedFloat(self.length(edge_id))).unwrap();
            let (a, b) = self.endpoints(longest_edge);
            self.splip_edge(a, b);
        }
    }
}
