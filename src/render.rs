use crate::dual::Orientation;
use crate::{to_color, to_principal_direction, vec3_to_vector3d, Configuration, InputResource, MainMesh, Perspective, RenderedMesh, SolutionResource};
use crate::{CameraHandles, PrincipalDirection};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy::render::camera::ScalingMode;
use bevy::render::render_resource::{Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages};
use douconel::douconel::Douconel;
use douconel::douconel_embedded::HasPosition;
use enum_iterator::{all, Sequence};
use hutspot::draw::DrawableLine;
use hutspot::geom::Vector3D;
use serde::{Deserialize, Serialize};
use slotmap::Key;
use smooth_bevy_cameras::controllers::orbit::{OrbitCameraBundle, OrbitCameraController};
use std::collections::HashMap;

const BACKGROUND_COLOR_SCREENSHOT_MODE: bevy::prelude::Color = bevy::prelude::Color::srgb(255. / 255., 255. / 255., 255. / 255.);
const BACKGROUND_COLOR: bevy::prelude::Color = bevy::prelude::Color::srgb(27. / 255., 27. / 255., 27. / 255.);
const DEFAULT_CAMERA_EYE: Vec3 = Vec3::new(25.0, 25.0, 35.0);
const DEFAULT_CAMERA_TARGET: Vec3 = Vec3::new(0., 0., 0.);
const DEFAULT_CAMERA_TEXTURE_SIZE: u32 = 640;

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone, Default, Serialize, Deserialize, Sequence)]
pub enum Objects {
    MeshDualLoops,
    #[default]
    PolycubeDual,
    PolycubePrimal,
    MeshPolycubeLayout,
    MeshAlignmentScore,
    Flag,
}

impl From<Objects> for String {
    fn from(val: Objects) -> Self {
        match val {
            Objects::MeshDualLoops => "dual loops",
            Objects::PolycubeDual => "polycube (dual)",
            Objects::PolycubePrimal => "polycube (primal)",
            Objects::MeshPolycubeLayout => "polycube segmentation",
            Objects::MeshAlignmentScore => "alignment (score)",
            Objects::Flag => "flag",
        }
        .to_owned()
    }
}

impl From<Objects> for Vec3 {
    fn from(val: Objects) -> Self {
        match val {
            Objects::MeshDualLoops => Self::new(0., 0., 0.),
            Objects::PolycubeDual => Self::new(1_000., 0., 0.),
            Objects::PolycubePrimal => Self::new(1_000., 1_000., 0.),
            Objects::MeshPolycubeLayout => Self::new(1_000., 1_000., 1_000.),
            Objects::MeshAlignmentScore => Self::new(0., 1_000., 0.),
            Objects::Flag => Self::new(0., 0., 1_000.),
        }
    }
}

#[derive(Component, PartialEq, Eq, Hash, Debug, Copy, Clone, Default, Serialize, Deserialize)]
pub struct CameraFor(pub Objects);

pub fn reset(
    commands: &mut Commands,
    cameras: &Query<Entity, With<Camera>>,
    images: &mut ResMut<Assets<Image>>,
    handles: &mut ResMut<CameraHandles>,
    configuration: &ResMut<Configuration>,
) {
    for camera in cameras.iter() {
        commands.entity(camera).despawn();
    }

    // Main camera. This is the camera that the user can control.
    commands
        .spawn(Camera3dBundle {
            camera: Camera {
                clear_color: ClearColorConfig::Custom(bevy::prelude::Color::srgb(
                    configuration.clear_color[0] as f32 / 255.,
                    configuration.clear_color[1] as f32 / 255.,
                    configuration.clear_color[2] as f32 / 255.,
                )),
                ..Default::default()
            },
            tonemapping: Tonemapping::None,
            ..default()
        })
        .insert((OrbitCameraBundle::new(
            OrbitCameraController {
                mouse_rotate_sensitivity: Vec2::splat(0.08),
                mouse_translate_sensitivity: Vec2::splat(0.1),
                mouse_wheel_zoom_sensitivity: 0.2,
                smoothing_weight: 0.8,
                ..Default::default()
            },
            DEFAULT_CAMERA_EYE + Vec3::from(Objects::MeshDualLoops),
            DEFAULT_CAMERA_TARGET + Vec3::from(Objects::MeshDualLoops),
            Vec3::Y,
        ),))
        .insert(CameraFor(Objects::MeshDualLoops));

    // Sub cameras. These cameras render to a texture.
    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size: Extent3d {
                width: DEFAULT_CAMERA_TEXTURE_SIZE,
                height: DEFAULT_CAMERA_TEXTURE_SIZE,
                ..default()
            },
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };
    image.resize(image.texture_descriptor.size);

    for object in all::<Objects>() {
        let handle = images.add(image.clone());
        handles.map.insert(CameraFor(object), handle.clone());
        let projection = if object == Objects::PolycubeDual || object == Objects::PolycubePrimal {
            bevy::prelude::Projection::Orthographic(OrthographicProjection {
                scaling_mode: ScalingMode::FixedVertical(30.0),
                ..default()
            })
        } else {
            bevy::prelude::Projection::default()
        };

        commands.spawn((
            Camera3dBundle {
                camera: Camera {
                    target: handle.into(),
                    clear_color: ClearColorConfig::Custom(bevy::prelude::Color::srgb(
                        configuration.clear_color[0] as f32 / 255.,
                        configuration.clear_color[1] as f32 / 255.,
                        configuration.clear_color[2] as f32 / 255.,
                    )),
                    ..Default::default()
                },
                projection,
                tonemapping: Tonemapping::None,
                ..default()
            },
            CameraFor(object),
        ));
    }
}

pub fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut handles: ResMut<CameraHandles>,
    mut config_store: ResMut<GizmoConfigStore>,
    cameras: Query<Entity, With<Camera>>,
    configuration: ResMut<Configuration>,
) {
    let (config, _) = config_store.config_mut::<DefaultGizmoConfigGroup>();
    config.line_width = 2.;

    self::reset(&mut commands, &cameras, &mut images, &mut handles, &configuration);
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct MeshProperties {
    pub source: String,
    pub scale: f32,
    pub translation: Vector3D,
}

fn get_pbrbundle(mesh: Handle<Mesh>, translation: Vec3, scale: f32, material: &Handle<StandardMaterial>) -> PbrBundle {
    PbrBundle {
        mesh,
        transform: Transform {
            translation,
            rotation: Quat::IDENTITY,
            scale: Vec3::splat(scale),
        },
        material: material.clone(),
        ..default()
    }
}

fn get_mesh<VertID: Key, V: Default + HasPosition, EdgeID: Key, E: Default, FaceID: Key, F: Default>(
    dcel: &Douconel<VertID, V, EdgeID, E, FaceID, F>,
    color_map: &HashMap<FaceID, [f32; 3]>,
) -> (Mesh, Vec3, f32) {
    let mesh = dcel.bevy(color_map);
    let aabb = mesh.compute_aabb().unwrap();
    let scale = 10. * (1. / aabb.half_extents.max_element());
    let translation = (-scale * aabb.center).into();
    (mesh, translation, scale)
}

pub fn update(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,

    configuration: Res<Configuration>,
    rendered_mesh_query: Query<Entity, With<RenderedMesh>>,

    mut mesh_resmut: ResMut<InputResource>,
    mut solution: ResMut<SolutionResource>,
    mut gizmos_cache: ResMut<GizmosCache>,
    mut cameras: Query<(&mut Transform, &mut Projection, &CameraFor)>,
) {
    let main_transform = cameras.iter().find(|(_, _, camera_for)| camera_for.0 == Objects::MeshDualLoops).unwrap().0;
    let normalized_translation = main_transform.translation - Vec3::from(Objects::MeshDualLoops);
    let normalized_rotation = main_transform.rotation;

    let distance = normalized_translation.length();

    for (mut sub_transform, mut sub_projection, sub_object) in &mut cameras {
        sub_transform.translation = normalized_translation + Vec3::from(sub_object.0);
        sub_transform.rotation = normalized_rotation;
        if let Projection::Orthographic(orthographic) = sub_projection.as_mut() {
            orthographic.scaling_mode = ScalingMode::FixedVertical(distance);
        }
    }

    // The rest of this function function should only be called when the mesh (RenderedMesh or Solution) is changed.
    if !mesh_resmut.is_changed() && !solution.is_changed() {
        return;
    }
    info!("InputResource or SolutionResource change has been detected. Updating all objects and gizmos.");

    for entity in rendered_mesh_query.iter() {
        commands.entity(entity).despawn();
    }
    info!("Objects despawned.");

    gizmos_cache.wireframe.clear();
    gizmos_cache.wireframe_granulated.clear();
    gizmos_cache.loops[0].clear();
    gizmos_cache.loops[1].clear();
    gizmos_cache.loops[2].clear();
    gizmos_cache.paths.clear();
    gizmos_cache.flat_edges.clear();
    info!("Gizmos cache cleared.");

    if mesh_resmut.mesh.faces.is_empty() {
        warn!("Current mesh is empty.");
        return;
    }

    let standard_material = materials.add(StandardMaterial { unlit: true, ..default() });
    let background_material = materials.add(StandardMaterial {
        base_color: bevy::prelude::Color::srgb(
            configuration.clear_color[0] as f32 / 255.,
            configuration.clear_color[1] as f32 / 255.,
            configuration.clear_color[2] as f32 / 255.,
        ),
        unlit: true,
        ..default()
    });

    for object in [Objects::MeshAlignmentScore, Objects::MeshDualLoops, Objects::MeshPolycubeLayout] {
        let (_, translation, scale) = get_mesh(&(*mesh_resmut.mesh).clone(), &HashMap::new());
        for edge_id in mesh_resmut.mesh.edge_ids() {
            let (u_id, v_id) = mesh_resmut.mesh.endpoints(edge_id);
            let u = mesh_resmut.mesh.position(u_id);
            let v = mesh_resmut.mesh.position(v_id);
            let n = mesh_resmut.mesh.edge_normal(edge_id);
            add_line2(
                &mut gizmos_cache.wireframe,
                u,
                v,
                n * 0.005,
                hutspot::color::GRAY,
                translation + Vec3::from(object),
                scale,
            );
        }
    }

    for object in [Objects::MeshAlignmentScore, Objects::MeshDualLoops, Objects::MeshPolycubeLayout] {
        if let Ok(layout) = solution.current_solution.layout.as_ref() {
            let (_, translation, scale) = get_mesh(&(layout.granulated_mesh).clone(), &HashMap::new());
            for edge_id in layout.granulated_mesh.edge_ids() {
                let (u_id, v_id) = layout.granulated_mesh.endpoints(edge_id);
                let u = layout.granulated_mesh.position(u_id);
                let v = layout.granulated_mesh.position(v_id);
                let n = layout.granulated_mesh.edge_normal(edge_id);
                add_line2(
                    &mut gizmos_cache.wireframe_granulated,
                    u,
                    v,
                    n * 0.005,
                    hutspot::color::GRAY,
                    translation + Vec3::from(object),
                    scale,
                );
            }
        }
    }

    for object in all::<Objects>() {
        match object {
            Objects::MeshDualLoops => {
                let (mesh, translation, scale) = get_mesh(&(*mesh_resmut.mesh).clone(), &HashMap::new());
                mesh_resmut.properties.scale = scale;
                mesh_resmut.properties.translation = vec3_to_vector3d(translation);

                commands.spawn((
                    get_pbrbundle(meshes.add(mesh), translation + Vec3::from(Objects::MeshDualLoops), scale, &standard_material),
                    RenderedMesh,
                    MainMesh,
                ));

                if let (Ok(dual), Ok(lay), Some(polycube)) = (
                    &solution.current_solution.dual,
                    &solution.current_solution.layout,
                    &solution.current_solution.polycube,
                ) {
                    // // draw a pointer to the first edge of each loop
                    // for (_, lewp) in &solution.current_solution.loops {
                    //     let first_edge = solution.current_solution.get_pairs_of_sequence(&lewp.edges).first().unwrap().to_owned();
                    //     let u = mesh_resmut.mesh.midpoint(first_edge[0]);
                    //     let v = mesh_resmut.mesh.midpoint(first_edge[1]);
                    //     let n = mesh_resmut.mesh.edge_normal(first_edge[0]);
                    //     add_line2(
                    //         &mut gizmos_cache.loops[0],
                    //         u,
                    //         v,
                    //         n * 0.01,
                    //         hutspot::color::RED,
                    //         translation + Vec3::from(Objects::MeshDualLoops),
                    //         scale,
                    //     );
                    // }

                    for segment_id in dual.loop_structure.edge_ids() {
                        let direction = dual.segment_to_direction(segment_id);
                        let orientation = dual.segment_to_orientation(segment_id);
                        if orientation == Orientation::Backwards {
                            continue;
                        }

                        let color = to_color(direction, Perspective::Dual, Some(Orientation::Forwards));
                        for [u, v] in solution.current_solution.get_pairs_of_sequence(&dual.segment_to_edges(segment_id)) {
                            add_line2(
                                &mut gizmos_cache.loops[direction as usize],
                                mesh_resmut.mesh.midpoint(u),
                                mesh_resmut.mesh.midpoint(v),
                                mesh_resmut.mesh.edge_normal(u) * 0.01,
                                color,
                                translation + Vec3::from(Objects::MeshDualLoops),
                                scale,
                            );
                        }

                        let color = to_color(direction, Perspective::Dual, Some(Orientation::Backwards));
                        for [u, v] in solution.current_solution.get_pairs_of_sequence(&dual.segment_to_edges(segment_id)) {
                            let direction_vector = mesh_resmut.mesh.midpoint(v) - mesh_resmut.mesh.midpoint(u);
                            let offset = mesh_resmut.mesh.edge_normal(u).cross(&direction_vector).normalize() * 0.05;

                            add_line2(
                                &mut gizmos_cache.loops[direction as usize],
                                mesh_resmut.mesh.midpoint(u),
                                mesh_resmut.mesh.midpoint(v),
                                mesh_resmut.mesh.edge_normal(u) * 0.01 + offset,
                                color,
                                translation + Vec3::from(Objects::MeshDualLoops),
                                scale,
                            );
                        }
                    }

                    for (&pedge_id, path) in &lay.edge_to_path {
                        let f1 = polycube.structure.normal(polycube.structure.face(pedge_id));
                        let f2 = polycube.structure.normal(polycube.structure.face(polycube.structure.twin(pedge_id)));
                        for vertexpair in path.windows(2) {
                            if lay.granulated_mesh.edge_between_verts(vertexpair[0], vertexpair[1]).is_none() {
                                println!("Edge between {:?} and {:?} does not exist", vertexpair[0], vertexpair[1]);
                                continue;
                            }
                            let edge_id = lay.granulated_mesh.edge_between_verts(vertexpair[0], vertexpair[1]).unwrap().0;
                            let (u_id, v_id) = lay.granulated_mesh.endpoints(edge_id);
                            let u = lay.granulated_mesh.position(u_id);
                            let v = lay.granulated_mesh.position(v_id);
                            let n = lay.granulated_mesh.edge_normal(edge_id);
                            if f1 == f2 {
                                add_line2(
                                    &mut gizmos_cache.flat_edges,
                                    u,
                                    v,
                                    n * 0.01,
                                    hutspot::color::GRAY,
                                    translation + Vec3::from(object),
                                    scale,
                                );
                            } else {
                                add_line2(
                                    &mut gizmos_cache.paths,
                                    u,
                                    v,
                                    n * 0.01,
                                    hutspot::color::GRAY,
                                    translation + Vec3::from(object),
                                    scale,
                                );
                            }
                        }
                    }
                } else {
                    for loop_id in solution.current_solution.loops.keys() {
                        let direction = solution.current_solution.loop_to_direction(loop_id);
                        let color = to_color(direction, Perspective::Dual, None);
                        for [u, v] in solution.current_solution.get_pairs_of_loop(loop_id) {
                            add_line2(
                                &mut gizmos_cache.loops[direction as usize],
                                mesh_resmut.mesh.midpoint(u),
                                mesh_resmut.mesh.midpoint(v),
                                mesh_resmut.mesh.edge_normal(u) * 0.01,
                                color,
                                translation + Vec3::from(Objects::MeshDualLoops),
                                scale,
                            );
                        }
                    }
                }
            }
            Objects::PolycubeDual => {
                if let Some(polycube) = &solution.current_solution.polycube {
                    let colormap = HashMap::new();
                    let (mesh, translation, scale) = get_mesh(&polycube.structure, &colormap);

                    commands.spawn((
                        get_pbrbundle(meshes.add(mesh), translation + Vec3::from(object), scale, &standard_material),
                        RenderedMesh,
                    ));

                    // Draw all loop segments per face.
                    for &face_id in &polycube.structure.face_ids() {
                        let this_centroid = polycube.structure.centroid(face_id);
                        let normal = (polycube.structure.normal(face_id) as Vector3D).normalize();
                        let direction = to_principal_direction(normal).0;

                        for &neighbor_id in &polycube.structure.fneighbors(face_id) {
                            let edge_between = polycube.structure.edge_between_faces(face_id, neighbor_id).unwrap().0;
                            let root = polycube.structure.root(edge_between);
                            let root_pos = polycube.structure.position(root);

                            let edge_between = polycube.structure.edge_between_faces(face_id, neighbor_id).unwrap().0;
                            let (vertex_start, vertex_end) = polycube.structure.endpoints(edge_between);
                            let (intersection_start, intersection_end) = (
                                polycube.region_to_vertex.get_by_right(&vertex_start).unwrap(),
                                polycube.region_to_vertex.get_by_right(&vertex_end).unwrap(),
                            );
                            let corresponding_segment = solution
                                .current_solution
                                .dual
                                .as_ref()
                                .unwrap()
                                .loop_structure
                                .edge_between_faces(*intersection_start, *intersection_end)
                                .unwrap()
                                .0;

                            let assigned_label = solution.current_solution.dual.as_ref().unwrap().segment_to_direction(corresponding_segment);

                            for orientation in [Orientation::Forwards, Orientation::Backwards] {
                                let segment_direction = match (direction, assigned_label) {
                                    (PrincipalDirection::X, PrincipalDirection::Y) | (PrincipalDirection::Y, PrincipalDirection::X) => PrincipalDirection::Z,
                                    (PrincipalDirection::X, PrincipalDirection::Z) | (PrincipalDirection::Z, PrincipalDirection::X) => PrincipalDirection::Y,
                                    (PrincipalDirection::Y, PrincipalDirection::Z) | (PrincipalDirection::Z, PrincipalDirection::Y) => PrincipalDirection::X,
                                    _ => unreachable!(),
                                };

                                let mut direction_vector = this_centroid;
                                direction_vector[segment_direction as usize] = root_pos[segment_direction as usize];

                                let mut offset = Vector3D::new(0., 0., 0.);

                                let dist = 0.001 * f64::from(scale);

                                let line = DrawableLine::from_line(
                                    this_centroid,
                                    direction_vector,
                                    offset + normal * 0.01,
                                    vec3_to_vector3d(translation + Vec3::from(object)),
                                    scale,
                                );
                                let c = to_color(assigned_label, Perspective::Dual, Some(orientation));
                                gizmos_cache.loops[assigned_label as usize].push((line.u, line.v, c));
                            }
                        }
                    }

                    // draw all edges of the polycube in white
                    if configuration.show_gizmos_paths {
                        for edge_id in polycube.structure.edge_ids() {
                            let endpoints = polycube.structure.endpoints(edge_id);
                            let u = polycube.structure.position(endpoints.0);
                            let v = polycube.structure.position(endpoints.1);
                            let f1 = polycube.structure.normal(polycube.structure.face(edge_id));
                            let f2 = polycube.structure.normal(polycube.structure.face(polycube.structure.twin(edge_id)));
                            let line = DrawableLine::from_line(u, v, ((f1 + f2) / 2.) * 0.05, vec3_to_vector3d(translation + Vec3::from(object)), scale);
                            if f1 == f2 {
                                gizmos_cache.flat_edges.push((line.u, line.v, hutspot::color::GRAY));
                            } else {
                                gizmos_cache.paths.push((line.u, line.v, hutspot::color::GRAY));
                            }
                        }
                    }
                }

                // if let Some(polycube) = &solution.current_solution.polycube.clone() {
                //     let (mesh, translation, scale) = get_mesh(&polycube.structure.clone(), &HashMap::new());

                //     commands.spawn((
                //         get_pbrbundle(meshes.add(mesh), translation + Vec3::from(object), scale, &standard_material),
                //         RenderedMesh,
                //     ));

                //     // Draw all loop segments / faces axis aligned.
                //     for &face_id in &polycube.structure.face_ids() {
                //         let original_id = polycube.intersection_to_face.get_by_right(&face_id).unwrap();
                //         let this_centroid = polycube.structure.centroid(face_id);

                //         let normal = (polycube.structure.normal(face_id) as Vector3D).normalize();
                //         let direction = to_principal_direction(normal).0;

                //         for &neighbor_id in &polycube.structure.fneighbors(face_id) {
                //             let next_original_id = polycube.intersection_to_face.get_by_right(&neighbor_id).unwrap();

                //             let edge_between = polycube.structure.edge_between_faces(face_id, neighbor_id).unwrap().0;
                //             let root = polycube.structure.root(edge_between);
                //             let root_pos = polycube.structure.position(root);

                //             let segment = solution
                //                 .current_solution
                //                 .dual
                //                 .as_ref()
                //                 .unwrap()
                //                 .loop_structure
                //                 .edge_between_verts(*original_id, *next_original_id)
                //                 .unwrap()
                //                 .0;

                //             let assigned_label = solution.current_solution.dual.as_ref().unwrap().segment_to_direction(segment);

                //             for orientation in [Orientation::Forwards, Orientation::Backwards] {
                //                 let segment_direction = match (orientation, direction) {
                //                     (PrincipalDirection::X, PrincipalDirection::Y) | (PrincipalDirection::Y, PrincipalDirection::X) => PrincipalDirection::Z,
                //                     (PrincipalDirection::X, PrincipalDirection::Z) | (PrincipalDirection::Z, PrincipalDirection::X) => PrincipalDirection::Y,
                //                     (PrincipalDirection::Y, PrincipalDirection::Z) | (PrincipalDirection::Z, PrincipalDirection::Y) => PrincipalDirection::X,
                //                     _ => unreachable!(),
                //                 };

                //                 let mut direction_vector = this_centroid;
                //                 direction_vector[segment_direction as usize] = root_pos[segment_direction as usize];

                //                 let mut offset = Vector3D::new(0., 0., 0.);

                //                 let dist = 0.001 * f64::from(scale);

                //                 match orientation {
                //                     Side::Upper => match direction {
                //                         PrincipalDirection::X => offset[0] += dist,
                //                         PrincipalDirection::Y => offset[1] += dist,
                //                         PrincipalDirection::Z => offset[2] += dist,
                //                     },
                //                     Side::Lower => match direction {
                //                         PrincipalDirection::X => offset[0] -= dist,
                //                         PrincipalDirection::Y => offset[1] -= dist,
                //                         PrincipalDirection::Z => offset[2] -= dist,
                //                     },
                //                 };

                //                 let line = DrawableLine::from_line(
                //                     this_centroid,
                //                     direction_vector,
                //                     offset + normal * 0.01,
                //                     vec3_to_vector3d(translation + Vec3::from(object)),
                //                     scale,
                //                 );
                //                 let c = to_color(direction, Perspective::Dual, Some(orientation));
                //                 gizmos_cache.lines.push((line.u, line.v, c));
                //             }
                //         }
                //     }

                //     // Draw the edges of the polycube.
                //     for edge_id in polycube.structure.edge_ids() {
                //         let endpoints = polycube.structure.endpoints(edge_id);
                //         let u = polycube.structure.position(endpoints.0);
                //         let v = polycube.structure.position(endpoints.1);
                //         let line = DrawableLine::from_line(
                //             u,
                //             v,
                //             polycube.structure.normal(polycube.structure.face(edge_id)) * 0.005,
                //             vec3_to_vector3d(translation + Vec3::from(object)),
                //             scale,
                //         );
                //         let c = hutspot::color::GRAY;
                //         gizmos_cache.lines.push((line.u, line.v, c));
                //     }
                // }
            }
            Objects::PolycubePrimal => {
                if let Some(polycube) = &solution.current_solution.polycube {
                    let mut colormap = HashMap::new();
                    for &face_id in &polycube.structure.face_ids() {
                        let normal = (polycube.structure.normal(face_id) as Vector3D).normalize();
                        let (dir, side) = to_principal_direction(normal);
                        let color = to_color(dir, Perspective::Primal, Some(side));
                        colormap.insert(face_id, color);
                    }

                    let (mesh, translation, scale) = get_mesh(&polycube.structure, &colormap);

                    commands.spawn((
                        get_pbrbundle(meshes.add(mesh), translation + Vec3::from(object), scale, &standard_material),
                        RenderedMesh,
                    ));

                    // Draw the edges of the polycube.
                    if configuration.show_gizmos_paths {
                        for edge_id in polycube.structure.edge_ids() {
                            let endpoints = polycube.structure.endpoints(edge_id);
                            let u = polycube.structure.position(endpoints.0);
                            let v = polycube.structure.position(endpoints.1);
                            let f1 = polycube.structure.normal(polycube.structure.face(edge_id));
                            let f2 = polycube.structure.normal(polycube.structure.face(polycube.structure.twin(edge_id)));
                            let line = DrawableLine::from_line(u, v, ((f1 + f2) / 2.) * 0.05, vec3_to_vector3d(translation + Vec3::from(object)), scale);
                            if f1 == f2 {
                                gizmos_cache.flat_edges.push((line.u, line.v, hutspot::color::BLACK));
                            } else {
                                gizmos_cache.paths.push((line.u, line.v, hutspot::color::BLACK));
                            }
                        }
                    }
                }
            }
            Objects::MeshPolycubeLayout => {
                if let Some(polycube) = &solution.current_solution.polycube {
                    if let Ok(lay) = &solution.current_solution.layout {
                        let mut layout_color_map = HashMap::new();
                        for &face_id in &polycube.structure.face_ids() {
                            let normal = (polycube.structure.normal(face_id) as Vector3D).normalize();
                            let (dir, side) = to_principal_direction(normal);
                            let color = to_color(dir, Perspective::Primal, Some(side));
                            for &triangle_id in &lay.face_to_patch[&face_id].faces {
                                layout_color_map.insert(triangle_id, color);
                            }
                        }

                        let (mesh, translation, scale) = get_mesh(&lay.granulated_mesh, &layout_color_map);

                        commands.spawn((
                            get_pbrbundle(meshes.add(mesh), translation + Vec3::from(object), scale, &standard_material),
                            RenderedMesh,
                        ));

                        for (&pedge_id, path) in &lay.edge_to_path {
                            let f1 = polycube.structure.normal(polycube.structure.face(pedge_id));
                            let f2 = polycube.structure.normal(polycube.structure.face(polycube.structure.twin(pedge_id)));
                            for vertexpair in path.windows(2) {
                                if lay.granulated_mesh.edge_between_verts(vertexpair[0], vertexpair[1]).is_none() {
                                    println!("Edge between {:?} and {:?} does not exist", vertexpair[0], vertexpair[1]);
                                    continue;
                                }
                                let edge_id = lay.granulated_mesh.edge_between_verts(vertexpair[0], vertexpair[1]).unwrap().0;
                                let (u_id, v_id) = lay.granulated_mesh.endpoints(edge_id);
                                let u = lay.granulated_mesh.position(u_id);
                                let v = lay.granulated_mesh.position(v_id);
                                let n = lay.granulated_mesh.edge_normal(edge_id);
                                if f1 == f2 {
                                    add_line2(
                                        &mut gizmos_cache.flat_edges,
                                        u,
                                        v,
                                        n * 0.01,
                                        hutspot::color::BLACK,
                                        translation + Vec3::from(object),
                                        scale,
                                    );
                                } else {
                                    add_line2(
                                        &mut gizmos_cache.paths,
                                        u,
                                        v,
                                        n * 0.01,
                                        hutspot::color::BLACK,
                                        translation + Vec3::from(object),
                                        scale,
                                    );
                                }
                            }
                        }
                    }
                }
            }
            Objects::MeshAlignmentScore => {
                if let Some(polycube) = &solution.current_solution.polycube {
                    if let Ok(lay) = &solution.current_solution.layout {
                        let mut layout_color_map = HashMap::new();

                        for &triangle_id in &lay.granulated_mesh.face_ids() {
                            let score = solution.current_solution.alignment_per_triangle[triangle_id];
                            let color = hutspot::color::map(score as f32, &hutspot::color::SCALE_MAGMA);
                            layout_color_map.insert(triangle_id, color);
                        }

                        let (mesh, translation, scale) = get_mesh(&lay.granulated_mesh, &layout_color_map);
                        commands.spawn((
                            get_pbrbundle(meshes.add(mesh), translation + Vec3::from(object), scale, &standard_material),
                            RenderedMesh,
                        ));

                        for (&pedge_id, path) in &lay.edge_to_path {
                            let f1 = polycube.structure.normal(polycube.structure.face(pedge_id));
                            let f2 = polycube.structure.normal(polycube.structure.face(polycube.structure.twin(pedge_id)));
                            for vertexpair in path.windows(2) {
                                if lay.granulated_mesh.edge_between_verts(vertexpair[0], vertexpair[1]).is_none() {
                                    println!("Edge between {:?} and {:?} does not exist", vertexpair[0], vertexpair[1]);
                                    continue;
                                }
                                let edge_id = lay.granulated_mesh.edge_between_verts(vertexpair[0], vertexpair[1]).unwrap().0;
                                let (u_id, v_id) = lay.granulated_mesh.endpoints(edge_id);
                                let u = lay.granulated_mesh.position(u_id);
                                let v = lay.granulated_mesh.position(v_id);
                                let n = lay.granulated_mesh.edge_normal(edge_id);
                                if f1 == f2 {
                                    add_line2(
                                        &mut gizmos_cache.flat_edges,
                                        u,
                                        v,
                                        n * 0.01,
                                        hutspot::color::BLACK,
                                        translation + Vec3::from(object),
                                        scale,
                                    );
                                } else {
                                    add_line2(
                                        &mut gizmos_cache.paths,
                                        u,
                                        v,
                                        n * 0.01,
                                        hutspot::color::BLACK,
                                        translation + Vec3::from(object),
                                        scale,
                                    );
                                }
                            }
                        }
                    }
                }
            }
            Objects::Flag => {
                if let Some(flags) = &solution.current_solution.external_flag {
                    let mut colormap = HashMap::new();

                    for (face_id, label) in flags {
                        let color = match label {
                            0 => hutspot::color::RED,
                            1 => hutspot::color::RED,
                            4 => hutspot::color::YELLOW,
                            5 => hutspot::color::YELLOW,
                            2 => hutspot::color::BLUE,
                            3 => hutspot::color::BLUE,
                            _ => hutspot::color::BLACK,
                        };
                        colormap.insert(face_id, color);
                    }

                    let (mesh, translation, scale) = get_mesh(&mesh_resmut.mesh, &colormap);

                    commands.spawn((
                        get_pbrbundle(meshes.add(mesh), translation + Vec3::from(object), scale, &standard_material),
                        RenderedMesh,
                    ));

                    for edge_id in mesh_resmut.mesh.edge_ids() {
                        let f1 = flags[mesh_resmut.mesh.face(edge_id)];
                        let f2 = flags[mesh_resmut.mesh.face(mesh_resmut.mesh.twin(edge_id))];
                        if f1 != f2 {
                            let (u_id, v_id) = mesh_resmut.mesh.endpoints(edge_id);
                            let u = mesh_resmut.mesh.position(u_id);
                            let v = mesh_resmut.mesh.position(v_id);
                            let n = mesh_resmut.mesh.edge_normal(edge_id);
                            add_line2(
                                &mut gizmos_cache.paths,
                                u,
                                v,
                                n * 0.01,
                                hutspot::color::BLACK,
                                translation + Vec3::from(object),
                                scale,
                            );
                        }
                    }
                }
            }
        }

        // Spawning covers such that the objects are view-blocked.
        commands.spawn((
            PbrBundle {
                mesh: meshes.add(Sphere::new(400.)),
                transform: Transform {
                    translation: Vec3::from(object),
                    ..default()
                },
                material: background_material.clone(),
                ..default()
            },
            RenderedMesh,
        ));
    }
}

#[derive(Default, Resource)]
pub struct GizmosCache {
    pub wireframe: Vec<Line>,
    pub wireframe_granulated: Vec<Line>,
    pub loops: [Vec<Line>; 3],
    pub paths: Vec<Line>,
    pub flat_edges: Vec<Line>,

    pub raycaster: Vec<Line>,
}

type Line = (Vec3, Vec3, hutspot::color::Color);

// Draws the gizmos. This includes all wireframes, vertices, normals, raycasts, etc.
pub fn gizmos(mut gizmos: Gizmos, gizmos_cache: Res<GizmosCache>, configuration: Res<Configuration>) {
    if configuration.show_gizmos_mesh {
        for &(u, v, c) in &gizmos_cache.wireframe {
            gizmos.line(u, v, Color::srgb(c[0], c[1], c[2]));
        }
    }

    if configuration.show_gizmos_mesh && configuration.show_gizmos_mesh_granulated {
        for &(u, v, c) in &gizmos_cache.wireframe_granulated {
            gizmos.line(u, v, Color::srgb(c[0], c[1], c[2]));
        }
    }

    if configuration.show_gizmos_loops[0] {
        for &(u, v, c) in &gizmos_cache.loops[0] {
            gizmos.line(u, v, Color::srgb(c[0], c[1], c[2]));
        }
    }

    if configuration.show_gizmos_loops[1] {
        for &(u, v, c) in &gizmos_cache.loops[1] {
            gizmos.line(u, v, Color::srgb(c[0], c[1], c[2]));
        }
    }

    if configuration.show_gizmos_loops[2] {
        for &(u, v, c) in &gizmos_cache.loops[2] {
            gizmos.line(u, v, Color::srgb(c[0], c[1], c[2]));
        }
    }

    if configuration.show_gizmos_paths {
        for &(u, v, c) in &gizmos_cache.paths {
            gizmos.line(u, v, Color::srgb(c[0], c[1], c[2]));
        }
    }

    if configuration.show_gizmos_paths && configuration.show_gizmos_flat_edges {
        for &(u, v, c) in &gizmos_cache.flat_edges {
            gizmos.line(u, v, Color::srgb(c[0], c[1], c[2]));
        }
    }

    if configuration.interactive {
        for &(u, v, c) in &gizmos_cache.raycaster {
            gizmos.line(u, v, Color::srgb(c[0], c[1], c[2]));
        }
    }
}

pub fn add_line(lines: &mut Vec<Line>, position: Vector3D, normal: Vector3D, length: f32, color: hutspot::color::Color, translation: Vector3D, scale: f32) {
    let line = DrawableLine::from_vertex(position, normal, length, translation, scale);
    lines.push((line.u, line.v, color));
}

pub fn add_line2(
    lines: &mut Vec<Line>,
    position_a: Vector3D,
    position_b: Vector3D,
    offset: Vector3D,
    color: hutspot::color::Color,
    translation: Vec3,
    scale: f32,
) {
    let line = DrawableLine::from_line(position_a, position_b, offset, vec3_to_vector3d(translation), scale);
    lines.push((line.u, line.v, color));
}
