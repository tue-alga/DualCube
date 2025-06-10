#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
pub mod douconel;
pub mod douconel_bevy;
pub mod douconel_embedded;
pub mod douconel_io;
pub mod douconel_petgraph;

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::{
        douconel::{Douconel, Empty},
        douconel_embedded::EmbeddedVertex,
    };

    slotmap::new_key_type! {
        struct VertID;
        struct EdgeID;
        struct FaceID;
    }

    #[test]
    fn from_manual() {
        let faces = vec![vec![0, 2, 1], vec![0, 1, 3], vec![1, 2, 3], vec![0, 3, 2]];
        let douconel = Douconel::<VertID, Empty, EdgeID, Empty, FaceID, Empty>::from_faces(&faces);
        assert!(douconel.is_ok(), "{douconel:?}");
        if let Ok((douconel, _, _)) = douconel {
            assert!(douconel.nr_verts() == 4);
            assert!(douconel.nr_edges() == 6 * 2);
            assert!(douconel.nr_faces() == 4);

            for face_id in douconel.faces.keys() {
                assert!(douconel.corners(face_id).len() == 3);
            }
        }
    }

    #[test]
    fn from_blub_stl() {
        let douconel = Douconel::<VertID, EmbeddedVertex, EdgeID, Empty, FaceID, Empty>::from_file(&PathBuf::from("assets/blub001k.stl"));
        assert!(douconel.is_ok(), "{douconel:?}");
        if let Ok((douconel, _, _)) = douconel {
            assert!(douconel.nr_verts() == 945);
            assert!(douconel.nr_edges() == 2829 * 2);
            assert!(douconel.nr_faces() == 1886);

            for face_id in douconel.faces.keys() {
                assert!(douconel.corners(face_id).len() == 3);
            }
        }
    }

    #[test]
    fn from_blub_obj() {
        let douconel = Douconel::<VertID, EmbeddedVertex, EdgeID, Empty, FaceID, Empty>::from_file(&PathBuf::from("assets/blub001k.obj"));
        assert!(douconel.is_ok(), "{douconel:?}");
        if let Ok((douconel, _, _)) = douconel {
            assert!(douconel.nr_verts() == 945);
            assert!(douconel.nr_edges() == 2829 * 2);
            assert!(douconel.nr_faces() == 1886);

            for face_id in douconel.faces.keys() {
                assert!(douconel.corners(face_id).len() == 3);
            }
        }
    }

    #[test]
    fn from_nefertiti_stl() {
        let douconel = Douconel::<VertID, EmbeddedVertex, EdgeID, Empty, FaceID, Empty>::from_file(&PathBuf::from("assets/nefertiti099k.stl"));
        assert!(douconel.is_ok(), "{douconel:?}");
        if let Ok((douconel, _, _)) = douconel {
            assert!(douconel.nr_verts() == 49971);
            assert!(douconel.nr_edges() == 149_907 * 2);
            assert!(douconel.nr_faces() == 99938);

            for face_id in douconel.faces.keys() {
                assert!(douconel.corners(face_id).len() == 3);
            }
        }
    }

    #[test]
    fn from_hexahedron_obj() {
        let douconel = Douconel::<VertID, EmbeddedVertex, EdgeID, Empty, FaceID, Empty>::from_file(&PathBuf::from("assets/hexahedron.obj"));
        assert!(douconel.is_ok(), "{douconel:?}");
        if let Ok((douconel, _, _)) = douconel {
            assert!(douconel.nr_verts() == 8);
            assert!(douconel.nr_edges() == 4 * 6);
            assert!(douconel.nr_faces() == 6);

            for face_id in douconel.faces.keys() {
                assert!(douconel.corners(face_id).len() == 4);
            }
        }
    }

    #[test]
    fn from_tetrahedron_obj() {
        let douconel = Douconel::<VertID, EmbeddedVertex, EdgeID, Empty, FaceID, Empty>::from_file(&PathBuf::from("assets/tetrahedron.obj"));
        assert!(douconel.is_ok(), "{douconel:?}");
        if let Ok((douconel, _, _)) = douconel {
            assert!(douconel.nr_verts() == 4);
            assert!(douconel.nr_edges() == 3 * 4);
            assert!(douconel.nr_faces() == 4);

            for face_id in douconel.faces.keys() {
                assert!(douconel.corners(face_id).len() == 3);
            }
        }
    }

    #[test]
    fn serialize() {
        let douconel = Douconel::<VertID, EmbeddedVertex, EdgeID, Empty, FaceID, Empty>::from_file(&PathBuf::from("assets/nefertiti099k.stl"));

        assert!(douconel.is_ok(), "{douconel:?}");
        if let Ok((douconel, _, _)) = douconel {
            println!("{:?}", douconel.verts);

            let serialized = serde_json::to_string(&douconel);
            assert!(serialized.is_ok(), "{:?}", serialized.unwrap());

            println!("{serialized:?}");

            if let Ok(serialized) = serialized {
                let deserialized = serde_json::from_str::<Douconel<VertID, EmbeddedVertex, EdgeID, Empty, FaceID, Empty>>(&serialized);

                assert!(deserialized.is_ok(), "{deserialized:?}");
                if let Ok(deserialized) = deserialized {
                    assert!(douconel.nr_verts() == deserialized.nr_verts());
                    assert!(douconel.nr_edges() == deserialized.nr_edges());
                    assert!(douconel.nr_faces() == deserialized.nr_faces());
                }
            }
        }
    }

    // #[test]
    // fn from_nefertiti_stl() {
    //     let douconel = Douconel::<EmbeddedVertex, (), ()>::from_stl("assets/nefertiti099k.stl");
    //     assert!(douconel.is_ok(), "{douconel:?}");
    //     if let Ok((douconel, _, _)) = douconel {
    //         assert!(douconel.nr_verts() == 49971);
    //         assert!(douconel.nr_edges() == 149907 * 2);
    //         assert!(douconel.nr_faces() == 99938);

    //         assert!(douconel.verify_properties().is_ok());
    //         assert!(douconel.verify_references().is_ok());
    //         assert!(douconel.verify_invariants().is_ok());

    //         for face_id in douconel.faces.keys() {
    //             assert!(douconel.corners(face_id).len() == 3);
    //         }

    //         let verts = douconel.verts.keys().collect_vec();

    //         const ITERATIONS: usize = 100;

    //         // hutspot::timer::bench(
    //         //     || {
    //         //         let mut rng = rand::thread_rng();
    //         //         let (v_a, v_b) = verts
    //         //             .choose_multiple(&mut rng, 2)
    //         //             .copied()
    //         //             .collect_tuple()
    //         //             .unwrap();

    //         //         let _path = petgraph::algo::astar(
    //         //             &g,
    //         //             v_a,
    //         //             |finish| finish == v_b,
    //         //             |e| *e.weight(),
    //         //             |v_id| douconel.distance(v_b, v_id),
    //         //         );
    //         //     },
    //         //     "petgraph_astar",
    //         //     ITERATIONS,
    //         // );

    //         // hutspot::timer::bench(
    //         //     || {
    //         //         let mut rng = rand::thread_rng();
    //         //         let (v_a, v_b) = verts
    //         //             .choose_multiple(&mut rng, 2)
    //         //             .copied()
    //         //             .collect_tuple()
    //         //             .unwrap();

    //         //         let _path = pathfinding::prelude::astar(
    //         //             &v_a,
    //         //             |&v_id| {
    //         //                 douconel
    //         //                     .vneighbors(v_id)
    //         //                     .iter()
    //         //                     .map(|&n_id| (n_id, OrderedFloat(douconel.distance(v_id, n_id))))
    //         //                     .collect_vec()
    //         //             },
    //         //             |&v_id| OrderedFloat(douconel.distance(v_id, v_b)),
    //         //             |&v_id| v_id == v_b,
    //         //         );
    //         //     },
    //         //     "pathfinding_astar",
    //         //     ITERATIONS,
    //         // );

    //         // let cache = Rc::new(RefCell::new(SecondaryMap::<
    //         //     VertID,
    //         //     Vec<(VertID, OrderedFloat<f32>)>,
    //         // >::new()));

    //         // hutspot::timer::bench(
    //         //     || {
    //         //         let mut rng = rand::thread_rng();
    //         //         let (v_a, v_b) = verts
    //         //             .choose_multiple(&mut rng, 2)
    //         //             .copied()
    //         //             .collect_tuple()
    //         //             .unwrap();

    //         //         douconel.find_shortest_path(v_a, v_b, cache.clone(), |a, b| {
    //         //             ordered_float::OrderedFloat(douconel.distance(a, b))
    //         //         });
    //         //     },
    //         //     "paths pathfinding_astar with cache",
    //         //     ITERATIONS,
    //         // );

    //         let mut cache = HashMap::<EdgeID, Vec<(EdgeID, OrderedFloat<f32>)>>::new();

    //         // hutspot::timer::bench(
    //         //     || {
    //         //         let mut rng = rand::thread_rng();
    //         //         let (v_a, v_b) = douconel
    //         //             .edges
    //         //             .keys()
    //         //             .collect_vec()
    //         //             .choose_multiple(&mut rng, 2)
    //         //             .copied()
    //         //             .collect_tuple()
    //         //             .unwrap();

    //         //         let _ = find_shortest_path(
    //         //             v_a,
    //         //             v_b,
    //         //             douconel.neighbor_function_edgegraph(),
    //         //             douconel.weight_function_angle_edges(2),
    //         //             &mut cache,
    //         //         );
    //         //     },
    //         //     "edgegraph",
    //         //     ITERATIONS,
    //         // );

    //         // let cache = Rc::new(RefCell::new(SecondaryMap::<
    //         //     VertID,
    //         //     Vec<(VertID, OrderedFloat<f32>)>,
    //         // >::new()));

    //         // hutspot::timer::bench(
    //         //     || {
    //         //         let mut rng = rand::thread_rng();
    //         //         let v_a = verts.choose(&mut rng).copied().unwrap();
    //         //         println!("{v_a:?}");
    //         //         let c = douconel.find_shortest_cycle(v_a, cache.clone(), |a, b| {
    //         //             ordered_float::OrderedFloat(douconel.distance(a, b))
    //         //         });
    //         //         println!("{c:?}");
    //         //     },
    //         //     "cycles pathfinding_astar with cache",
    //         //     ITERATIONS,
    //         // );
    //     }
    // }
}
