use crate::{
    graph::Graaf,
    solutions::{Loop, LoopID},
    EdgeID, EmbeddedMesh, PrincipalDirection, VertID,
};
use douconel::douconel::Douconel;
use itertools::Itertools;
use log::error;
use slotmap::{SecondaryMap, SlotMap};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

#[derive(Default, Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Orientation {
    #[default]
    Forwards,
    Backwards,
}

// A collection of loops forms a loop structure; a graph, where
// the vertices correspond to loop intersections,
// the edges correspond to loop segments,
// and the faces correspond to loop regions.
slotmap::new_key_type! {
    pub struct LoopIntersectionID;
    pub struct LoopSegmentID;
    pub struct LoopRegionID;
    pub struct ZoneID;
}

#[derive(Default, Copy, Clone, Debug)]
pub struct LoopSegment {
    // A loop segment has a corresponding loop (id) and an orientation (either following the direction of the loop, or opposite direction of the loop)
    pub loop_id: LoopID,
    // TODO: make a function that reads the orientation by checking the corresponding loop..
    pub orientation: Orientation,
}

#[derive(Default, Clone, Debug)]
pub struct LoopRegion {
    // A loop region has a corresponding surface, in this implementation, the surface is defined by a set of mesh vertices
    pub verts: HashSet<VertID>,
}

#[derive(Clone, Debug)]
pub struct Zone {
    // A zone is defined by a direction
    pub direction: PrincipalDirection,
    // All regions that are part of the zone
    pub regions: HashSet<LoopRegionID>,
}

#[derive(Default, Clone, Debug)]
pub struct LevelGraphs {
    //
    pub zones: SlotMap<ZoneID, Zone>,
    //
    pub graphs: [Graaf<ZoneID, LoopID>; 3],
    //
    pub region_to_zones: [SecondaryMap<LoopRegionID, ZoneID>; 3],
    //
    pub levels: [Vec<HashSet<ZoneID>>; 3],
}

pub type LoopStructure = Douconel<LoopIntersectionID, EdgeID, LoopSegmentID, LoopSegment, LoopRegionID, LoopRegion>;

// Dual structure (of a polycube)
#[derive(Debug, Clone)]
pub struct Dual {
    pub mesh_ref: Arc<EmbeddedMesh>,
    // TODO: make this an actual (arc) reference somehow?
    pub loops_ref: SlotMap<LoopID, Loop>,

    pub loop_structure: LoopStructure,
    pub level_graphs: LevelGraphs,
}

#[derive(Default, Debug, Clone)]
pub enum PropertyViolationError {
    #[default]
    UnknownError,
    FaceWithDegreeLessThanThree,
    FaceWithDegreeMoreThanSix,
    InvalidFaceBoundary,
    CyclicDependency,
    PathEmpty,
    LoopHasTooFewIntersections,
}

impl Dual {
    pub fn from(mesh_ref: Arc<EmbeddedMesh>, loops_ref: &SlotMap<LoopID, Loop>) -> Result<Self, PropertyViolationError> {
        let mut dual = Self {
            mesh_ref,
            loops_ref: loops_ref.clone(),
            loop_structure: Douconel::default(),
            level_graphs: LevelGraphs::default(),
        };

        // Find all intersections and loop regions induced by the loops, and compute the loop structure
        dual.assign_loop_structure()?;

        // For each loop region, find its actual subsurface (on the mesh)
        dual.assign_subsurfaces()?;

        // Find the zones and construct the level graphs
        dual.assign_level_graphs();

        // Verify properties
        dual.verify_properties()?;

        // Assign levels to the loop structure
        dual.assign_levels();

        Ok(dual)
    }

    pub fn intersection_to_edge(&self, intersection: LoopIntersectionID) -> EdgeID {
        self.loop_structure.verts[intersection]
    }

    pub fn segment_to_loop(&self, segment: LoopSegmentID) -> LoopID {
        self.loop_structure.edges[segment].loop_id
    }

    pub fn segment_to_orientation(&self, segment: LoopSegmentID) -> Orientation {
        self.loop_structure.edges[segment].orientation
    }

    pub fn segment_to_endpoints(&self, segment: LoopSegmentID) -> (EdgeID, EdgeID) {
        let endpoints = &self.loop_structure.endpoints(segment);
        (self.intersection_to_edge(endpoints.0), self.intersection_to_edge(endpoints.1))
    }

    pub fn region_to_zone(&self, region: LoopRegionID, direction: PrincipalDirection) -> ZoneID {
        self.level_graphs.region_to_zones[direction as usize][region]
    }

    pub fn segment_to_edges_excl(&self, segment: LoopSegmentID) -> Vec<EdgeID> {
        let inclusive_edges = self.segment_to_edges_incl(segment);
        inclusive_edges[2..inclusive_edges.len() - 2].to_vec()
    }

    pub fn segment_to_edges_incl(&self, segment: LoopSegmentID) -> Vec<EdgeID> {
        let (start, end) = self.segment_to_endpoints(segment);
        let (start_twin, end_twin) = (self.mesh_ref.twin(start), self.mesh_ref.twin(end));
        let loop_id = self.segment_to_loop(segment);
        if self.segment_to_orientation(segment) == Orientation::Forwards {
            let mut edges = self.loops_ref[loop_id].between(start, end);
            if !edges.contains(&start_twin) {
                edges.insert(0, start_twin);
            }
            if !edges.contains(&end_twin) {
                edges.push(end_twin);
            }
            edges
        } else {
            let mut edges = self.loops_ref[loop_id].between(end, start);
            if !edges.contains(&start_twin) {
                edges.push(start_twin);
            }
            if !edges.contains(&end_twin) {
                edges.insert(0, end_twin);
            }
            edges
        }
    }

    pub fn segment_to_edges(&self, segment: LoopSegmentID) -> Vec<EdgeID> {
        let edges = self.segment_to_edges_incl(segment);

        let mut fixed_edges = vec![];

        for edge_pair in edges.windows(2) {
            let (from, to) = (edge_pair[0], edge_pair[1]);

            // they are either twins, or they share a face
            let they_are_twins = self.mesh_ref.twin(from) == to;
            let they_share_face = self.mesh_ref.face(from) == self.mesh_ref.face(to);

            if they_are_twins || they_share_face {
                fixed_edges.push(from);
            } else {
                // there is an edge missing between them.
                // this missing edge is either the twin of from or the twin of to.
                let candidate_missing1 = self.mesh_ref.twin(from);
                let candidate_missing2 = self.mesh_ref.twin(to);

                // the true missing edge is the one that is twin to one and shares face with the other.
                let candidate_missing1_is_true = self.mesh_ref.face(candidate_missing1) == self.mesh_ref.face(to);
                let candidate_missing2_is_true = self.mesh_ref.face(candidate_missing2) == self.mesh_ref.face(from);
                assert!(candidate_missing1_is_true ^ candidate_missing2_is_true);

                let missing = if candidate_missing1_is_true { candidate_missing1 } else { candidate_missing2 };

                fixed_edges.push(from);
                fixed_edges.push(missing);
            }
        }
        fixed_edges.push(edges[edges.len() - 1]);

        fixed_edges
    }

    pub fn segment_to_direction(&self, segment: LoopSegmentID) -> PrincipalDirection {
        let loop_id = self.segment_to_loop(segment);
        self.loops_ref[loop_id].direction
    }

    fn pos<T: PartialEq>(list: &[T], needle: &T) -> usize {
        list.iter().position(|x| x == needle).unwrap()
    }

    fn next<T: PartialEq + Copy>(list: &[T], needle: T) -> T {
        let pos = Self::pos(list, &needle);
        list[(pos + 1) % list.len()]
    }

    fn prev<T: PartialEq + Copy>(list: &[T], needle: T) -> T {
        let pos = Self::pos(list, &needle);
        list[(pos + list.len() - 1) % list.len()]
    }

    // Returns error if:
    //    1. A loop has less than 4 intersections.
    //    2. A face has more than 6 edges (we know the face degree is at most 6, so we can early stop, and we also want to prevent infinite loops / malformed faces)
    //    3. Invalid intersection.
    fn assign_loop_structure(&mut self) -> Result<(), PropertyViolationError> {
        // For each edge, we store the loops that pass it
        let mut occupied: HashMap<EdgeID, Vec<LoopID>> = HashMap::new();
        for loop_id in self.loops_ref.keys() {
            for &edge in &self.loops_ref[loop_id].edges {
                occupied.entry(edge).or_default().push(loop_id);
            }
        }

        // Intersections are edges that are occupied exactly twice. It is not possible for an edge to be occupied more than twice.
        // NOTE: An intersection exists on two half-edges, we only store the intersection at the lower ID half-edge
        if occupied.values().any(|x| x.len() >= 3) {
            // error!("Invalid intersection: an edge is occupied by more than two loops.");
            return Err(PropertyViolationError::UnknownError);
        }

        let intersection_markers: HashMap<EdgeID, [LoopID; 2]> = occupied
            .into_iter()
            .filter_map(|(edge, loops)| {
                assert!(loops.len() <= 2);
                if loops.len() == 2 && edge > self.mesh_ref.twin(edge) {
                    Some((edge, [loops[0], loops[1]]))
                } else {
                    None
                }
            })
            .collect();

        // For each loop we find its intersections
        let loop_to_intersection_markers: HashMap<LoopID, Vec<EdgeID>> = self
            .loops_ref
            .iter()
            .map(|(loop_id, lewp)| {
                (
                    loop_id,
                    lewp.edges.iter().filter(|edge| intersection_markers.contains_key(edge)).copied().collect_vec(),
                )
            })
            .collect();

        // If any loop has too few intersections (less than 4), we return an error
        if loop_to_intersection_markers.values().any(|x| x.len() < 4) {
            return Err(PropertyViolationError::LoopHasTooFewIntersections);
        }

        // For each intersection we find its adjacent intersections (should be 4, by following its associated (two) loops in all (two) directions.
        let mut intersections = HashMap::new();
        for (intersection_id, [l1, l2]) in intersection_markers {
            let this_edge = intersection_id;
            let twin_edge = self.mesh_ref.twin(this_edge);
            let quad = self.mesh_ref.quad(this_edge);

            // Find the adjacent intersections in l1
            let l1_next_intersection = Self::next(&loop_to_intersection_markers[&l1], this_edge);
            let l1_prev_intersection = Self::prev(&loop_to_intersection_markers[&l1], this_edge);

            // Find the adjacent intersections in l2
            let l2_next_intersection = Self::next(&loop_to_intersection_markers[&l2], this_edge);
            let l2_prev_intersection = Self::prev(&loop_to_intersection_markers[&l2], this_edge);

            let mut l1_edges_prev = Self::prev(&self.loops_ref[l1].edges, this_edge);
            let mut l1_edges_next = Self::next(&self.loops_ref[l1].edges, this_edge);
            // Either the prev or next is the twin edge, go one step further.
            if l1_edges_prev == twin_edge {
                l1_edges_prev = Self::prev(&self.loops_ref[l1].edges, l1_edges_prev);
            } else if l1_edges_next == twin_edge {
                l1_edges_next = Self::next(&self.loops_ref[l1].edges, l1_edges_next);
            }

            let mut l2_edges_prev = Self::prev(&self.loops_ref[l2].edges, this_edge);
            let mut l2_edges_next = Self::next(&self.loops_ref[l2].edges, this_edge);
            // Either the prev or next is the twin edge, go one step further.
            if l2_edges_prev == twin_edge {
                l2_edges_prev = Self::prev(&self.loops_ref[l2].edges, l2_edges_prev);
            } else if l2_edges_next == twin_edge {
                l2_edges_next = Self::next(&self.loops_ref[l2].edges, l2_edges_next);
            }

            // We can order the intersections based on the local ordering of the edges in the loops
            let ordered_adjacent_intersections = quad
                .iter()
                .filter_map(|&x| match x {
                    x if x == l1_edges_next => Some((l1, l1_next_intersection, Orientation::Forwards)),
                    x if x == l1_edges_prev => Some((l1, l1_prev_intersection, Orientation::Backwards)),
                    x if x == l2_edges_next => Some((l2, l2_next_intersection, Orientation::Forwards)),
                    x if x == l2_edges_prev => Some((l2, l2_prev_intersection, Orientation::Backwards)),
                    _ => None,
                })
                .collect_vec();
            if (ordered_adjacent_intersections.len() != 4) || (ordered_adjacent_intersections.iter().map(|x| x.1).collect::<HashSet<_>>().len() != 4) {
                // error!(
                //     "Invalid intersection: ordered adjacent intersections are not unique or not 4. {:?} ({:?})",
                //     ordered_adjacent_intersections, quad
                // );
                return Err(PropertyViolationError::UnknownError);
            }

            assert!(ordered_adjacent_intersections.len() == 4);
            assert!(ordered_adjacent_intersections.iter().map(|x| x.1).collect::<HashSet<_>>().len() == 4);

            if ordered_adjacent_intersections[0].0 == ordered_adjacent_intersections[1].0 {
                // error!("[0].0 == [1].0: {:?}", ordered_adjacent_intersections[0].0);
                return Err(PropertyViolationError::UnknownError);
            }
            assert!(ordered_adjacent_intersections[0].0 != ordered_adjacent_intersections[1].0);

            if ordered_adjacent_intersections[1].0 == ordered_adjacent_intersections[2].0 {
                // error!("[1].0 == [2].0: {:?}", ordered_adjacent_intersections[1].0);
                return Err(PropertyViolationError::UnknownError);
            }
            assert!(ordered_adjacent_intersections[1].0 != ordered_adjacent_intersections[2].0);

            if ordered_adjacent_intersections[2].0 == ordered_adjacent_intersections[3].0 {
                // error!("[2].0 == [3].0: {:?}", ordered_adjacent_intersections[2].0);
                return Err(PropertyViolationError::UnknownError);
            }
            assert!(ordered_adjacent_intersections[2].0 != ordered_adjacent_intersections[3].0);

            if ordered_adjacent_intersections[3].0 == ordered_adjacent_intersections[0].0 {
                // error!("[3].0 == [0].0: {:?}", ordered_adjacent_intersections[3].0);
                return Err(PropertyViolationError::UnknownError);
            }
            assert!(ordered_adjacent_intersections[3].0 != ordered_adjacent_intersections[0].0);

            // Add the four adjacent intersections
            intersections.insert(
                this_edge,
                [
                    ordered_adjacent_intersections[0],
                    ordered_adjacent_intersections[1],
                    ordered_adjacent_intersections[2],
                    ordered_adjacent_intersections[3],
                ],
            );
        }

        // Create DCEL based on the intersections and loop regions
        // Construct all faces
        let edge_id_to_index: HashMap<EdgeID, usize> = intersections.keys().enumerate().map(|(i, &e)| (e, i)).collect();

        let mut edges = intersections
            .iter()
            .flat_map(|(&this, nexts)| nexts.iter().map(move |next| (this, next.1)))
            .collect_vec();

        let mut faces = vec![];
        while let Some(start) = edges.pop() {
            let mut counter = 0;
            let mut face = vec![start.0, start.1];
            loop {
                let u = face[face.len() - 2];
                let v = face[face.len() - 1];
                // get all intersections that are adjacent to v
                let adj = intersections[&v];
                let u_index = adj.iter().position(|&(_, x, _)| x == u).unwrap();
                let w = adj[(u_index + 4 - 1) % 4].1;
                edges.retain(|e| !(e.0 == v && e.1 == w));
                if w == face[0] {
                    break;
                }
                counter += 1;
                if counter > 6 {
                    return Err(PropertyViolationError::FaceWithDegreeMoreThanSix);
                }
                face.push(w);
            }
            faces.push(face.iter().map(|&x| edge_id_to_index[&x]).collect_vec());
        }

        if let Ok((mut douconel, vmap, _)) = LoopStructure::from_faces(&faces) {
            assert!(4 * douconel.vert_ids().len() == douconel.edge_ids().len());
            let intersection_ids = intersections.keys().copied().collect_vec();
            for vertex_id in douconel.vert_ids() {
                douconel.verts[vertex_id] = intersection_ids[vmap.get_by_right(&vertex_id).unwrap().to_owned()];
            }
            for edge_id in douconel.edge_ids() {
                let this = douconel.verts[douconel.root(edge_id)];
                let next = douconel.verts[douconel.toor(edge_id)];
                let (loop_id, _, orientation) = intersections[&this].iter().find(|&(_, x, _)| *x == next).unwrap().to_owned();
                douconel.edges[edge_id] = LoopSegment { loop_id, orientation }
            }

            self.loop_structure = douconel;
        } else {
            // error!("Failed to create loop structure from faces.");
            return Err(PropertyViolationError::UnknownError);
        }

        Ok(())
    }

    fn assign_subsurfaces(&mut self) -> Result<(), PropertyViolationError> {
        // All edges contained in loops can be considered blocked
        let blocked = self.loops_ref.values().flat_map(|lewp| lewp.edges.iter().copied()).collect::<HashSet<_>>();
        // Then all connected components of the mesh that are not blocked are loop regions
        let loop_regions = hutspot::graph::find_ccs(&self.mesh_ref.verts.keys().collect_vec(), |vertex| {
            let mut neighbors = self.mesh_ref.vneighbors(vertex);
            neighbors.retain(|&neighbor| !blocked.contains(&self.mesh_ref.edge_between_verts(vertex, neighbor).unwrap().0));
            neighbors
        });
        // This number should be equal to the number of faces in the loop structure
        if loop_regions.len() != self.loop_structure.face_ids().len() {
            // error!(
            //     "Invalid number of loop regions: expected {}, got {}",
            //     self.loop_structure.face_ids().len(),
            //     loop_regions.len()
            // );
            return Err(PropertyViolationError::UnknownError);
        }

        // Every loop segment should be part of exactly TWO connected components (on both sides)
        let mut segment_to_components: HashMap<LoopSegmentID, [usize; 2]> = HashMap::new();
        for &segment_id in &self.loop_structure.edge_ids() {
            // Loop segment should simply have only two connected components (one for each side)
            // We do not check all its edges, but only the first one (since they should all be the same)
            let arbitrary_edge = self.segment_to_edges_excl(segment_id)[0];
            let (vert1, vert2) = self.mesh_ref.endpoints(arbitrary_edge);
            let component1 = loop_regions.iter().position(|cc| cc.contains(&vert1)).unwrap();
            let component2 = loop_regions.iter().position(|cc| cc.contains(&vert2)).unwrap();
            segment_to_components.insert(segment_id, [component1, component2]);
        }

        // For every loop region, get the connected component that is shared among its loop segments
        for &face_id in &self.loop_structure.face_ids() {
            let loop_segments = self.loop_structure.edges(face_id);
            // Select an arbitrary loop segment
            let [component1, component2] = segment_to_components[&loop_segments[0]];
            // Check whether all loop segments share the same connected component
            let component1_is_shared = loop_segments.iter().all(|&segment| segment_to_components[&segment].contains(&component1));
            let component2_is_shared = loop_segments.iter().all(|&segment| segment_to_components[&segment].contains(&component2));
            self.loop_structure.faces[face_id] = LoopRegion {
                verts: match (component1_is_shared, component2_is_shared) {
                    (true, false) => loop_regions[component1].clone(),
                    (false, true) => loop_regions[component2].clone(),
                    _ => panic!(),
                },
            };
        }

        Ok(())
    }

    fn assign_level_graphs(&mut self) {
        // A zone is a collection of loop regions that are connected, and are bounded by only one type of loop segment (either X, Y, or Z).
        self.level_graphs.zones = SlotMap::with_key();

        for direction in [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z] {
            // All loop segments with the given direction are blocked
            let blocked = self
                .loop_structure
                .edge_ids()
                .into_iter()
                .filter(|&segment| self.segment_to_direction(segment) == direction)
                .collect::<HashSet<_>>();

            // Then all connected components of the loop structure that are not blocked are zones
            let zones = hutspot::graph::find_ccs(&self.loop_structure.face_ids(), |loop_region_id| {
                let mut neighbors = self.loop_structure.fneighbors(loop_region_id);
                neighbors.retain(|&neighbor_id| !blocked.contains(&self.loop_structure.edge_between_faces(loop_region_id, neighbor_id).unwrap().0));
                neighbors
            });
            for zone in zones {
                self.level_graphs.zones.insert(Zone { direction, regions: zone });
            }
        }

        for direction in [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z] {
            // Add all the zones as nodes to the level graph
            let mut edges = vec![];

            let zone_ids = self
                .level_graphs
                .zones
                .iter()
                .filter(|(_, zone)| zone.direction == direction)
                .map(|(zone_id, _)| zone_id)
                .collect_vec();

            // Create a mapping from loop regions to zones
            let region_to_zone = zone_ids
                .iter()
                .flat_map(|&zone_id| self.level_graphs.zones[zone_id].regions.iter().map(move |&region_id| (region_id, zone_id)))
                .collect::<SecondaryMap<_, _>>();
            self.level_graphs.region_to_zones[direction as usize] = region_to_zone;

            for &zone_id in &zone_ids {
                // The loop segments (in direction) of this zone
                let segments = self.level_graphs.zones[zone_id]
                    .regions
                    .iter()
                    .flat_map(|&region_id| self.loop_structure.edges(region_id))
                    .filter(|&segment_id| {
                        self.segment_to_direction(segment_id) == direction && self.segment_to_orientation(segment_id) == Orientation::Forwards
                    });
                // The adjacent loop regions of this zone
                let adjacent_regions = segments.map(|segment_id| {
                    let corresponding_loop = self.segment_to_loop(segment_id);
                    (self.loop_structure.face(self.loop_structure.twin(segment_id)), corresponding_loop)
                });
                // The adjacent zones of this zone
                let adjacent_zones = adjacent_regions.map(|(region_id, corresponding_loop)| (self.region_to_zone(region_id, direction), corresponding_loop));

                for (adjacent_id, loop_id) in adjacent_zones {
                    edges.push((zone_id, adjacent_id, loop_id));
                }

                self.level_graphs.graphs[direction as usize] = Graaf::from(zone_ids.clone(), edges.clone());
            }
        }
    }

    pub fn assign_levels(&mut self) {
        for direction in [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z] {
            let graph = &self.level_graphs.graphs[direction as usize];
            let start = graph.nodes().first().unwrap().to_owned();

            let mut queue = vec![start];
            let mut visited: HashSet<ZoneID> = HashSet::default();
            visited.insert(start);
            let mut levels = HashMap::new();
            levels.insert(start, 100_000usize);

            while let Some(node) = queue.pop() {
                let node_level = levels.get(&node).unwrap().to_owned();
                for neighbor in graph.neighbors_undirected(node) {
                    if visited.contains(&neighbor) {
                        continue;
                    }
                    match (graph.directed_edge_exists(node, neighbor), graph.directed_edge_exists(neighbor, node)) {
                        (true, false) => {
                            levels.insert(neighbor, node_level + 1);
                        }
                        (false, true) => {
                            levels.insert(neighbor, node_level - 1);
                        }
                        _ => {
                            panic!();
                        }
                    }
                    visited.insert(neighbor);
                    queue.push(neighbor);
                }
            }

            let mut level_to_zone = HashMap::new();
            let minimum = *levels.values().min().unwrap();
            for (zone_id, level) in levels {
                level_to_zone.insert(zone_id, level - minimum);
            }
            self.level_graphs.levels[direction as usize] = vec![HashSet::new(); *level_to_zone.values().max().unwrap() + 1];
            for (zone_id, level) in level_to_zone {
                self.level_graphs.levels[direction as usize][level].insert(zone_id);
            }
        }
    }

    fn verify_properties(&self) -> Result<(), PropertyViolationError> {
        // Definition 3.2. An oriented loop structure L is a polycube loop structure if:
        // 1. No three loops intersect at a single point.
        // 2. Each loop region is bounded by at least three loop segments.
        // 3. Within each loop region boundary, no two loop segments have the same axis label and side label.
        // 4. Each loop region has the topology of a disk.
        // 5. The level graphs are acyclic.

        // 1. is verified by construction, simply by the way we construct the loop structure.

        for face_id in self.loop_structure.face_ids() {
            let edges = self.loop_structure.edges(face_id);

            // Verify 2.
            if edges.len() < 3 {
                return Err(PropertyViolationError::FaceWithDegreeLessThanThree);
            }

            // Verify 3.
            let mut label_count = [0; 6];
            for edge in edges {
                let loop_id = self.segment_to_loop(edge);
                let direction = self.loops_ref[loop_id].direction;
                let orientation = self.loop_structure.edges[edge].orientation;
                match (direction, orientation) {
                    (PrincipalDirection::X, Orientation::Forwards) => label_count[0] += 1,
                    (PrincipalDirection::X, Orientation::Backwards) => label_count[1] += 1,
                    (PrincipalDirection::Y, Orientation::Forwards) => label_count[2] += 1,
                    (PrincipalDirection::Y, Orientation::Backwards) => label_count[3] += 1,
                    (PrincipalDirection::Z, Orientation::Forwards) => label_count[4] += 1,
                    (PrincipalDirection::Z, Orientation::Backwards) => label_count[5] += 1,
                }
            }
            if label_count.iter().any(|&x| x > 1) {
                return Err(PropertyViolationError::InvalidFaceBoundary);
            }
        }

        // 4. must be verified: TODO

        // Verify 5.
        for graph in &self.level_graphs.graphs {
            let topological_sort = hutspot::graph::topological_sort::<ZoneID>(&graph.nodes(), |z| graph.neighbors(z));
            if topological_sort.is_none() {
                return Err(PropertyViolationError::CyclicDependency);
            }
        }

        Ok(())
    }
}
