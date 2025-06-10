mod dual;
mod graph;
mod layout;
mod polycube;
mod render;
mod solutions;
mod ui;

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{block_on, AsyncComputeTaskPool, Task};
use bevy::window::WindowMode;
use bevy::winit::WinitWindows;
use bevy_egui::EguiPlugin;
use bevy_mod_raycast::prelude::*;
use douconel::douconel::Douconel;
use douconel::{douconel::Empty, douconel_embedded::EmbeddedVertex};
use dual::Orientation;
use graph::Graaf;
use hutspot::consts::{EPS, PI};
use hutspot::geom::Vector3D;
use itertools::Itertools;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use ordered_float::OrderedFloat;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use render::{add_line, add_line2, CameraFor, GizmosCache, MeshProperties, Objects};
use serde::{Deserialize, Serialize};
use slotmap::SecondaryMap;
use smooth_bevy_cameras::controllers::orbit::OrbitCameraPlugin;
use smooth_bevy_cameras::LookTransformPlugin;
use solutions::{Loop, Solution};
use std::collections::HashMap;
use std::env;
use std::fmt::Display;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
// use winit::platform::windows::WindowExtWindows;
use winit::window::Icon;

slotmap::new_key_type! {
    pub struct VertID;
    pub struct EdgeID;
    pub struct FaceID;
}

// Principal directions, used to characterize a polycube (each edge and face is associated with a principal direction)
#[derive(Copy, Clone, Default, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub enum PrincipalDirection {
    #[default]
    X,
    Y,
    Z,
}

impl Display for PrincipalDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::X => write!(f, "Z-axis"),
            Self::Y => write!(f, "Y-axis"),
            Self::Z => write!(f, "X-axis"),
        }
    }
}

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub enum Perspective {
    Primal,
    #[default]
    Dual,
}

pub const fn to_color(direction: PrincipalDirection, perspective: Perspective, orientation: Option<Orientation>) -> hutspot::color::Color {
    match (perspective, direction, orientation) {
        (Perspective::Primal, PrincipalDirection::X, None) => hutspot::color::RED,
        (Perspective::Primal, PrincipalDirection::Y, None) => hutspot::color::BLUE,
        (Perspective::Primal, PrincipalDirection::Z, None) => hutspot::color::YELLOW,
        (Perspective::Primal, PrincipalDirection::X, Some(Orientation::Forwards)) => hutspot::color::RED,
        (Perspective::Primal, PrincipalDirection::X, Some(Orientation::Backwards)) => hutspot::color::RED,
        (Perspective::Primal, PrincipalDirection::Y, Some(Orientation::Forwards)) => hutspot::color::BLUE,
        (Perspective::Primal, PrincipalDirection::Y, Some(Orientation::Backwards)) => hutspot::color::BLUE,
        (Perspective::Primal, PrincipalDirection::Z, Some(Orientation::Forwards)) => hutspot::color::YELLOW,
        (Perspective::Primal, PrincipalDirection::Z, Some(Orientation::Backwards)) => hutspot::color::YELLOW,
        (Perspective::Dual, PrincipalDirection::X, None) => hutspot::color::GREEN,
        (Perspective::Dual, PrincipalDirection::Y, None) => hutspot::color::ORANGE,
        (Perspective::Dual, PrincipalDirection::Z, None) => hutspot::color::PURPLE,
        (Perspective::Dual, PrincipalDirection::X, Some(Orientation::Forwards)) => hutspot::color::GREEN,
        (Perspective::Dual, PrincipalDirection::X, Some(Orientation::Backwards)) => hutspot::color::GREEN_LIGHT,
        (Perspective::Dual, PrincipalDirection::Y, Some(Orientation::Forwards)) => hutspot::color::ORANGE,
        (Perspective::Dual, PrincipalDirection::Y, Some(Orientation::Backwards)) => hutspot::color::ORANG_LIGHT,
        (Perspective::Dual, PrincipalDirection::Z, Some(Orientation::Forwards)) => hutspot::color::PURPLE,
        (Perspective::Dual, PrincipalDirection::Z, Some(Orientation::Backwards)) => hutspot::color::PURPLE_LIGHT,
    }
}

impl From<PrincipalDirection> for Vector3D {
    fn from(dir: PrincipalDirection) -> Self {
        match dir {
            PrincipalDirection::X => Self::new(1., 0., 0.),
            PrincipalDirection::Y => Self::new(0., 1., 0.),
            PrincipalDirection::Z => Self::new(0., 0., 1.),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Side {
    Upper,
    Lower,
}

pub fn to_principal_direction(v: Vector3D) -> (PrincipalDirection, Orientation) {
    let x_is_max = v.x.abs() >= v.y.abs() && v.x.abs() >= v.z.abs();
    let y_is_max = v.y.abs() > v.x.abs() && v.y.abs() >= v.z.abs();
    let z_is_max = v.z.abs() > v.x.abs() && v.z.abs() > v.y.abs();
    assert!(x_is_max ^ y_is_max ^ z_is_max, "{v:?}");

    if x_is_max {
        if v.x > 0. {
            (PrincipalDirection::X, Orientation::Forwards)
        } else {
            (PrincipalDirection::X, Orientation::Backwards)
        }
    } else if y_is_max {
        if v.y > 0. {
            (PrincipalDirection::Y, Orientation::Forwards)
        } else {
            (PrincipalDirection::Y, Orientation::Backwards)
        }
    } else if z_is_max {
        if v.z > 0. {
            (PrincipalDirection::Z, Orientation::Forwards)
        } else {
            (PrincipalDirection::Z, Orientation::Backwards)
        }
    } else {
        unreachable!()
    }
}

pub type EmbeddedMesh = Douconel<VertID, EmbeddedVertex, EdgeID, Empty, FaceID, Empty>;

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
pub struct Configuration {
    pub direction: PrincipalDirection,
    pub alpha: f64,

    pub should_continue: bool,

    pub fps: f64,
    pub raycasted: Option<[EdgeID; 2]>,
    pub selected: Option<[EdgeID; 2]>,

    pub automatic: bool,
    pub interactive: bool,

    pub ui_is_hovered: [bool; 32],
    pub window_shows_object: [Objects; 4],
    pub window_has_size: [f32; 4],
    pub window_has_position: [(f32, f32); 4],

    pub hex_mesh_status: HexMeshStatus,
    pub smoothen_status: ActionEventStatus,

    pub show_gizmos_mesh: bool,
    pub show_gizmos_mesh_granulated: bool,
    pub show_gizmos_loops: [bool; 3],
    pub show_gizmos_paths: bool,
    pub show_gizmos_flat_edges: bool,

    pub clear_color: [u8; 3],

    pub unit_cubes: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum HexMeshStatus {
    None,
    Loading,
    Done(HexMeshScore),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HexMeshScore {
    hausdorff: f32,
    avg_jacob: f32,
    min_jacob: f32,
    max_jacob: f32,
    irregular: f32,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            direction: PrincipalDirection::X,
            alpha: 0.5,
            should_continue: false,
            fps: -1.,
            raycasted: None,
            selected: None,
            automatic: false,
            interactive: false,
            ui_is_hovered: [false; 32],
            window_shows_object: [
                Objects::MeshPolycubeLayout,
                Objects::PolycubePrimal,
                Objects::MeshAlignmentScore,
                Objects::PolycubeDual,
            ],
            window_has_size: [256., 256., 256., 0.],
            window_has_position: [(0., 0.); 4],
            hex_mesh_status: HexMeshStatus::None,
            smoothen_status: ActionEventStatus::None,
            show_gizmos_mesh: false,
            show_gizmos_mesh_granulated: false,
            show_gizmos_loops: [true, true, true],
            show_gizmos_paths: true,
            show_gizmos_flat_edges: false,
            clear_color: [27, 27, 27],
            unit_cubes: true,
        }
    }
}

// Updates the FPS counter in `configuration`.
pub fn fps(diagnostics: Res<DiagnosticsStore>, mut configuration: ResMut<Configuration>) {
    configuration.fps = -1.;
    if let Some(value) = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(bevy::diagnostic::Diagnostic::smoothed)
    {
        configuration.fps = value;
    }
}

#[derive(Resource, Default)]
struct Tasks {
    generating_chunks: HashMap<ActionEvent, Task<Option<Solution>>>,
}

#[derive(Resource, Default)]
struct HexTasks {
    generating_chunks: HashMap<usize, Task<Option<HexMeshScore>>>,
}

#[derive(Resource, Default)]
pub struct CameraHandles {
    pub map: HashMap<CameraFor, Handle<Image>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SaveStateObject {
    mesh: EmbeddedMesh,
    loops: Vec<Loop>,
    configuration: Configuration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SaveStateObjectBackwardscompatibility {
    pub mesh: EmbeddedMesh,
    pub loops: Vec<Loop>,
}

#[derive(Component)]
pub struct RenderedMesh;

#[derive(Component)]
pub struct MainMesh;

#[derive(Event, Debug, Eq, Hash, PartialEq)]
pub enum ActionEvent {
    LoadFile(PathBuf),
    ExportAll,
    ExportState,
    ExportSolution,
    ExportNLR,
    ToHexmesh,
    ResetCamera,
    Mutate,
    Smoothen,
    Recompute,
    Initialize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ActionEventStatus {
    None,
    Loading,
    Done(String),
}

// implement default for KdTree using the New Type Idiom
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TreeD(KdTree<f64, VertID, [f64; 3]>);
impl TreeD {
    fn nearest(&self, point: &[f64; 3]) -> (f64, VertID) {
        let neighbors = self.0.nearest(point, 1, &squared_euclidean).unwrap();
        let (d, i) = neighbors.first().unwrap();
        (*d, **i)
    }

    fn add(&mut self, point: [f64; 3], index: VertID) {
        self.0.add(point, index).unwrap();
    }
}
impl Default for TreeD {
    fn default() -> Self {
        Self(KdTree::new(3))
    }
}

#[derive(Default, Resource)]
pub struct CacheResource {
    cache: [HashMap<[EdgeID; 2], Vec<([EdgeID; 2], OrderedFloat<f64>)>>; 3],
}

#[derive(Default, Debug, Clone, Resource, Serialize, Deserialize)]
pub struct InputResource {
    mesh: Arc<EmbeddedMesh>,
    properties: MeshProperties,
    properties2: MeshProperties,
    vertex_lookup: TreeD,
    flow_graphs: [Graaf<EdgeID, f64>; 3],
}

#[derive(Debug, Clone, Resource)]
pub struct SolutionResource {
    current_solution: Solution,
    next: [HashMap<[EdgeID; 2], Option<Solution>>; 3],
}

impl Default for SolutionResource {
    fn default() -> Self {
        Self {
            current_solution: Solution::new(Arc::new(Douconel::default())),
            next: [HashMap::new(), HashMap::new(), HashMap::new()],
        }
    }
}

fn main() {
    App::new()
        .init_resource::<InputResource>()
        .init_resource::<CacheResource>()
        .init_resource::<GizmosCache>()
        .init_resource::<SolutionResource>()
        .init_resource::<Configuration>()
        .init_resource::<Tasks>()
        .init_resource::<HexTasks>()
        .init_resource::<CameraHandles>()
        .insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 1.0,
        })
        // Load default plugins
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "DualCube".to_string(),
                mode: WindowMode::Windowed,
                ..Default::default()
            }),
            ..Default::default()
        }))
        // Cursor ray
        .add_plugins(CursorRayPlugin)
        // Plugin for FPS
        .add_plugins(FrameTimeDiagnosticsPlugin)
        // Plugin for GUI
        .add_plugins(EguiPlugin)
        // Plugin for smooth camera
        .add_plugins(LookTransformPlugin)
        .add_plugins(OrbitCameraPlugin::default())
        // Setups
        .add_systems(Startup, ui::setup)
        .add_systems(Startup, render::setup)
        .add_systems(Startup, set_window_icon)
        // Updates
        .add_systems(Update, ui::update)
        .add_systems(Update, render::update)
        .add_systems(Update, render::gizmos)
        .add_systems(Update, handle_events)
        .add_systems(Update, handle_solution_tasks)
        .add_systems(Update, handle_hexmesh_tasks)
        .add_systems(Update, raycast)
        .add_systems(Update, fps)
        // .add_systems(Update, handle_tasks)
        .add_event::<ActionEvent>()
        .run();
}

fn set_window_icon(windows: NonSend<WinitWindows>) {
    let window = windows.windows.iter().next().unwrap().1;
    let set_icon = |path: &str, set_fn: fn(&winit::window::Window, Option<Icon>)| {
        let image = image::open(path).expect("Failed to open icon path").into_rgba8();
        set_fn(window, Some(Icon::from_rgba(image.clone().into_raw(), image.width(), image.height()).unwrap()));
    };
    set_icon("assets/logo-32.png", winit::window::Window::set_window_icon);
    // set_icon("assets/logo-512.png", winit::window::Window::set_taskbar_icon);
}

pub fn handle_hexmesh_tasks(mut tasks: ResMut<HexTasks>, mut configuration: ResMut<Configuration>) {
    tasks.generating_chunks.retain(|id, task| {
        // check on our task to see how it's doing :)
        let status = block_on(future::poll_once(task));

        // keep the entry in our HashMap only if the task is not done yet
        let retain = status.is_none();

        // if this task is done, handle the data it returned!
        if let Some(chunk_data) = status {
            println!("HexTask {} is done!", id);

            if let Some(label) = chunk_data {
                configuration.hex_mesh_status = HexMeshStatus::Done(label);
            }
        }

        retain
    });
}

pub fn handle_solution_tasks(
    mut tasks: ResMut<Tasks>,
    mut solution: ResMut<SolutionResource>,
    mut ev_writer: EventWriter<ActionEvent>,
    mut configuration: ResMut<Configuration>,
) {
    tasks.generating_chunks.retain(|id, task| {
        // check on our task to see how it's doing :)
        let status = block_on(future::poll_once(task));

        // if this task is done, handle the data it returned!
        if let Some(chunk_data) = status {
            match &id {
                ActionEvent::Mutate => {
                    if let Some(new_solution) = chunk_data {
                        if let Some(quality) = new_solution.get_quality() {
                            if quality >= solution.current_solution.get_quality().unwrap() {
                                println!("New solution is better than current solution. Updating...");
                                solution.current_solution = new_solution;
                            }
                        }
                    }
                }
                ActionEvent::Smoothen => {
                    if let Some(new_solution) = chunk_data {
                        println!("Smoothened solution. Updating...");
                        solution.current_solution = new_solution;
                    }

                    configuration.smoothen_status = ActionEventStatus::Done("Smoothening done.".to_string());
                }
                ActionEvent::Recompute => {
                    if let Some(new_solution) = chunk_data {
                        println!("Recomputed solution. Updating...");
                        solution.current_solution = new_solution;
                    }
                }
                _ => {}
            }

            false
        } else {
            true
        }
    });
}

pub fn handle_events(
    mut ev_reader: EventReader<ActionEvent>,
    mut mesh_resmut: ResMut<InputResource>,
    mut solution: ResMut<SolutionResource>,
    mut configuration: ResMut<Configuration>,

    mut tasks: ResMut<Tasks>,
    mut hextasks: ResMut<HexTasks>,

    mut commands: Commands,
    cameras: Query<Entity, With<Camera>>,

    mut images: ResMut<Assets<Image>>,
    mut camera_handles: ResMut<CameraHandles>,
) {
    for ev in ev_reader.read() {
        info!("Received event {ev:?}. Handling...");
        match ev {
            ActionEvent::LoadFile(path) => {
                match path.extension().unwrap().to_str() {
                    Some("obj" | "stl") => {
                        let current_configuration = (
                            configuration.window_has_position,
                            configuration.window_has_size,
                            configuration.window_shows_object,
                            configuration.clear_color,
                        );

                        *mesh_resmut = InputResource::default();
                        *solution = SolutionResource::default();

                        mesh_resmut.mesh = match Douconel::from_file(path) {
                            Ok(res) => {
                                let mut mesh = res.0;
                                Arc::new(mesh)
                            }
                            Err(err) => {
                                panic!("Error while parsing STL file {path:?}: {err:?}");
                            }
                        };

                        solution.current_solution = Solution::new(mesh_resmut.mesh.clone());
                    }
                    Some("save") => {
                        let current_configuration = (
                            configuration.window_has_position,
                            configuration.window_has_size,
                            configuration.window_shows_object,
                            configuration.clear_color,
                        );

                        *mesh_resmut = InputResource::default();
                        *solution = SolutionResource::default();

                        if let Ok(loaded_state) = serde_json::from_reader::<_, SaveStateObject>(BufReader::new(File::open(path).unwrap())) {
                            mesh_resmut.mesh = Arc::new(loaded_state.mesh);
                            solution.current_solution = Solution::new(mesh_resmut.mesh.clone());
                            *configuration = loaded_state.configuration.clone();

                            // overwrite
                            configuration.window_has_position = current_configuration.0;
                            configuration.window_has_size = current_configuration.1;
                            configuration.window_shows_object = current_configuration.2;
                            configuration.clear_color = current_configuration.3;

                            for saved_loop in loaded_state.loops {
                                solution.current_solution.add_loop(saved_loop);
                            }
                        } else if let Ok(loaded_state) =
                            serde_json::from_reader::<_, SaveStateObjectBackwardscompatibility>(BufReader::new(File::open(path).unwrap()))
                        {
                            mesh_resmut.mesh = Arc::new(loaded_state.mesh);
                            solution.current_solution = Solution::new(mesh_resmut.mesh.clone());

                            for saved_loop in loaded_state.loops {
                                solution.current_solution.add_loop(saved_loop);
                            }
                        } else {
                            println!("Error while parsing save file {path:?}");
                        }
                    }
                    Some("flag") => {
                        let mut flags = SecondaryMap::new();

                        for (i, line) in BufReader::new(File::open(path).unwrap()).lines().flatten().enumerate() {
                            let label = line.parse::<usize>().unwrap();
                            let face_id = mesh_resmut.mesh.face_ids()[i];
                            flags.insert(face_id, label);
                        }

                        solution.current_solution.external_flag = Some(flags);
                    }
                    _ => panic!("File format not supported."),
                }

                configuration.automatic = false;
                configuration.interactive = false;

                if mesh_resmut.mesh.vert_ids().is_empty() {
                    return;
                }

                configuration.hex_mesh_status = HexMeshStatus::None;

                let mut patterns = TreeD::default();
                for v_id in mesh_resmut.mesh.vert_ids() {
                    patterns.add(mesh_resmut.mesh.position(v_id).into(), v_id);
                }
                mesh_resmut.vertex_lookup = patterns;

                mesh_resmut.properties.source = String::from(path.to_str().unwrap());

                let nodes = mesh_resmut.mesh.edge_ids();

                for axis in [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z] {
                    let edges = nodes
                        .clone()
                        .into_par_iter()
                        .flat_map(|node| {
                            mesh_resmut.mesh.neighbor_function_edgegraph()(node)
                                .into_iter()
                                .map(|neighbor| {
                                    let face1 = mesh_resmut.mesh.face(node);
                                    let face2 = mesh_resmut.mesh.face(neighbor);

                                    if face1 == face2 {
                                        let normal = mesh_resmut.mesh.normal(face1);
                                        let m1 = mesh_resmut.mesh.midpoint(node);
                                        let m2 = mesh_resmut.mesh.midpoint(neighbor);
                                        let direction = m2 - m1;
                                        let cross = direction.cross(&normal);
                                        let angle = cross.angle(&axis.into());

                                        (node, neighbor, angle)
                                    } else {
                                        assert!(mesh_resmut.mesh.twin(node) == neighbor);
                                        (node, neighbor, 0.)
                                    }
                                })
                                .collect_vec()
                        })
                        .collect::<Vec<_>>();

                    mesh_resmut.flow_graphs[axis as usize] = Graaf::from(nodes.clone(), edges);
                }

                solution.current_solution.reconstruct_solution(configuration.unit_cubes);
            }
            ActionEvent::ExportNLR => {
                let cur = format!("{}", env::current_dir().unwrap().clone().display());

                let t = SystemTime::now().duration_since(UNIX_EPOCH).expect("Time went backwards").as_secs();
                let n = mesh_resmut
                    .properties
                    .source
                    .split("\\")
                    .last()
                    .unwrap()
                    .split('.')
                    .next()
                    .unwrap()
                    .split(' ')
                    .next()
                    .unwrap();

                let path_topol = format!("{cur}/out/{t}.topol");
                let path_geom = format!("{cur}/out/{t}.geom",);
                let path_cdim = format!("{cur}/out/{t}.cdim",);

                solution
                    .current_solution
                    .export_to_nlr(&PathBuf::from(path_topol), &PathBuf::from(path_geom), &PathBuf::from(path_cdim));
            }
            ActionEvent::ExportAll => {
                if mesh_resmut.mesh.vert_ids().is_empty() {
                    return;
                }

                let t = SystemTime::now().duration_since(UNIX_EPOCH).expect("Time went backwards").as_secs();
                let n = mesh_resmut
                    .properties
                    .source
                    .split("\\")
                    .last()
                    .unwrap()
                    .split('.')
                    .next()
                    .unwrap()
                    .split(' ')
                    .next()
                    .unwrap();

                // TODO: fix name

                let path_save = format!("./out/{t}.save");
                let path_obj = format!("./out/{t}.obj",);
                let path_flag = format!("./out/{t}.flag",);

                let state = SaveStateObject {
                    mesh: (*mesh_resmut.mesh).clone(),
                    loops: solution.current_solution.loops.values().cloned().collect(),
                    configuration: configuration.clone(),
                };

                fs::write(&PathBuf::from(path_save), serde_json::to_string(&state).unwrap());

                solution.current_solution.export(&PathBuf::from(path_obj), &PathBuf::from(path_flag));
            }
            ActionEvent::ExportState => {
                if mesh_resmut.mesh.vert_ids().is_empty() {
                    return;
                }

                let cur = format!("{}", env::current_dir().unwrap().clone().display());

                let t = SystemTime::now().duration_since(UNIX_EPOCH).expect("Time went backwards").as_secs();
                let n = mesh_resmut
                    .properties
                    .source
                    .split("\\")
                    .last()
                    .unwrap()
                    .split('.')
                    .next()
                    .unwrap()
                    .split(' ')
                    .next()
                    .unwrap();

                let path_save = format!("{cur}/out/{t}.save",);

                let state = SaveStateObject {
                    mesh: (*mesh_resmut.mesh).clone(),
                    loops: solution.current_solution.loops.values().cloned().collect(),
                    configuration: configuration.clone(),
                };

                info!("writing to {cur}/out/{n}_{t}.save");

                let res = fs::write(PathBuf::from(path_save), serde_json::to_string(&state).unwrap());
                info!("{:?}", res);
            }
            ActionEvent::ExportSolution => {
                if mesh_resmut.mesh.vert_ids().is_empty() {
                    return;
                }

                let t = SystemTime::now().duration_since(UNIX_EPOCH).expect("Time went backwards").as_secs();
                let n = mesh_resmut
                    .properties
                    .source
                    .split("\\")
                    .last()
                    .unwrap()
                    .split('.')
                    .next()
                    .unwrap()
                    .split(' ')
                    .next()
                    .unwrap();

                let path_obj = format!("./out/temp/{n}_{t:?}.obj");
                let path_flag = format!("./out/temp/{n}_{t:?}.flag");

                solution.current_solution.export(&PathBuf::from(path_obj), &PathBuf::from(path_flag));
            }
            ActionEvent::ToHexmesh => {
                if mesh_resmut.mesh.vert_ids().is_empty() {
                    return;
                }

                let t = SystemTime::now().duration_since(UNIX_EPOCH).expect("Time went backwards").as_secs();
                let n = mesh_resmut.properties.source.split("\\").last().unwrap().split('.').next().unwrap().to_owned();

                let path_obj = format!("./out/temp/{n}_{t}.obj",);
                let path_flag = format!("./out/temp/{n}_{t}.flag",);
                let path_save = format!("./out/temp/{n}_{t}.save");

                let state = SaveStateObject {
                    mesh: (*mesh_resmut.mesh).clone(),
                    loops: solution.current_solution.loops.values().cloned().collect(),
                    configuration: configuration.clone(),
                };

                fs::write(&PathBuf::from(path_save), serde_json::to_string(&state).unwrap());

                solution.current_solution.export(&PathBuf::from(path_obj), &PathBuf::from(path_flag));

                configuration.hex_mesh_status = HexMeshStatus::Loading;

                let task_pool = AsyncComputeTaskPool::get();
                let task = task_pool.spawn(async move {
                    let out = Command::new("wsl")
                        .args([
                            "python3",
                            "~/polycube-to-hexmesh/seg_eval.py",
                            &format!("/mnt/c/Users/20182085/Documents/DualCube/out/temp/{n}_{t}.obj"),
                            &format!("/mnt/c/Users/20182085/Documents/DualCube/out/temp/{n}_{t}.flag"),
                        ])
                        .output()
                        .expect("failed to execute process");

                    println!("Seg output: {:?}", out);

                    let out = Command::new("wsl")
                        .args([
                            "~/polycube-to-hexmesh/pipeline.sh",
                            &format!("/mnt/c/Users/20182085/Documents/DualCube/out/temp/{n}_{t}.obj"),
                            "-algo",
                            &format!("/mnt/c/Users/20182085/Documents/DualCube/out/temp/{n}_{t}.flag"),
                        ])
                        .output()
                        .expect("failed to execute process");

                    println!("Pipeline output: {:?}", out);

                    // outputs to '~/polycube-to-hexmesh/out/{n}_{t}_hex.mesh'

                    let out = Command::new("wsl")
                        .args([
                            "python3",
                            "~/polycube-to-hexmesh/evaluator_old.py",
                            &format!("/home/snoep/polycube-to-hexmesh/tmp/{n}_{t}_custom_remesh.mesh"),
                            &format!("/home/snoep/polycube-to-hexmesh/out/{n}_{t}_custom_hex.mesh"),
                        ])
                        .output()
                        .expect("failed to execute process");

                    println!("Hexmesh evaluation output: {:?}", out);

                    let out_str = String::from_utf8(out.stdout).unwrap();
                    let out_vec = out_str.split('\n').filter(|s| !s.is_empty()).collect_vec();

                    let score = HexMeshScore {
                        hausdorff: out_vec[0].parse().unwrap(),
                        avg_jacob: out_vec[1].parse().unwrap(),
                        min_jacob: out_vec[2].parse().unwrap(),
                        max_jacob: out_vec[3].parse().unwrap(),
                        irregular: out_vec[4].parse().unwrap(),
                    };

                    Some(score)
                });

                hextasks.generating_chunks.insert(999, task);
            }
            ActionEvent::ResetCamera => {
                render::reset(&mut commands, &cameras, &mut images, &mut camera_handles, &configuration);
            }
            ActionEvent::Mutate => {
                let task_pool = AsyncComputeTaskPool::get();

                let cloned_solution = solution.current_solution.clone();
                let cloned_flow_graphs = mesh_resmut.flow_graphs.clone();

                let pool_size = 10;
                let pool2_size = 30;

                let task = task_pool.spawn(async move {
                    let mut pool1 = vec![(cloned_solution.clone(), cloned_solution.get_quality().unwrap()); pool_size];

                    for _ in 0..10 {
                        let pool2 = (0..pool2_size)
                            .into_par_iter()
                            .map(|_| {
                                // Grab a random solution from pool1
                                let mut rng = rand::thread_rng();
                                let index = rand::Rng::gen_range(&mut rng, 0..pool1.len());
                                pool1[index].clone()
                            })
                            .filter_map(|(sol, _)| {
                                // Mutate the solution
                                sol.mutation(&cloned_flow_graphs).map_or_else(
                                    || None,
                                    |mutation| {
                                        let quality = mutation.get_quality().unwrap_or(0.0);
                                        // Compute quality
                                        Some((mutation, quality))
                                    },
                                )
                            })
                            .collect::<Vec<_>>()
                            .into_iter()
                            .sorted_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .rev();

                        for (_, quality) in pool2.clone() {
                            println!("Generated solution with quality: {quality}");
                        }

                        // overwrite pool1 with top 5 solutions of pool1 and top 5 solutions of pool2

                        pool1 = pool1
                            .into_iter()
                            .take(5)
                            .chain(pool2.take(5))
                            .sorted_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .rev()
                            .collect();

                        for (_, quality) in &pool1 {
                            println!("PICKED solution with quality: {quality}");
                        }
                    }

                    if pool1.is_empty() {
                        println!("No valid solutions generated.");
                        return None;
                    }

                    // Log all qualities of generated solutions
                    for (sol, quality) in &pool1 {
                        println!("Generated solution with quality: {quality}");
                    }

                    // Grab the best solution
                    let (sol, quality) = pool1.into_iter().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
                    println!("Picked solution with quality: {quality}");
                    Some(sol)
                });

                tasks.generating_chunks.insert(ActionEvent::Mutate, task);
            }

            ActionEvent::Smoothen => {
                let cloned_sol = solution.current_solution.clone();
                tasks.generating_chunks.insert(
                    ActionEvent::Smoothen,
                    AsyncComputeTaskPool::get().spawn(async move {
                        let mut new_sol = cloned_sol;
                        if let Ok(lay) = new_sol.layout.as_mut() {
                            lay.smoothen();
                            lay.verify_paths();
                            lay.assign_patches();
                        }
                        new_sol.compute_quality();
                        Some(new_sol)
                    }),
                );
            }
            ActionEvent::Recompute => {
                println!("Once");
                let mut cloned_solution = solution.current_solution.clone();
                let unit = configuration.unit_cubes;
                tasks.generating_chunks.insert(
                    ActionEvent::Recompute,
                    AsyncComputeTaskPool::get().spawn(async move {
                        {
                            cloned_solution.resize_polycube(unit);
                            Some(cloned_solution)
                        }
                    }),
                );
            }
            ActionEvent::Initialize => {
                solution.current_solution.initialize(&mesh_resmut.flow_graphs);
            }
        }
    }
}

#[inline]
fn vec3_to_vector3d(v: Vec3) -> Vector3D {
    Vector3D::new(v.x.into(), v.y.into(), v.z.into())
}

fn raycast(
    cursor_ray: Res<CursorRay>,
    mut raycast: Raycast,
    mut mouse: ResMut<ButtonInput<MouseButton>>,
    mut evr_motion: EventReader<MouseMotion>,
    mut evr_scroll: EventReader<MouseWheel>,
    mut keyboard: ResMut<ButtonInput<KeyCode>>,
    mesh_resmut: Res<InputResource>,
    mut solution: ResMut<SolutionResource>,
    mut cache: ResMut<CacheResource>,
    mut gizmos_cache: ResMut<GizmosCache>,
    mut configuration: ResMut<Configuration>,
    query: Query<Entity, With<MainMesh>>,
) {
    configuration.raycasted = None;
    configuration.selected = None;
    gizmos_cache.raycaster.clear();
    if !configuration.interactive {
        return;
    }

    // Render all current solutions  (for currently selected direction)
    for (&edgepair, sol) in &solution.next[configuration.direction as usize] {
        let u = mesh_resmut.mesh.midpoint(edgepair[0]);
        let v = mesh_resmut.mesh.midpoint(edgepair[1]);
        let n = mesh_resmut.mesh.normal(mesh_resmut.mesh.face(edgepair[0]));
        let color = match sol {
            Some(_) => to_color(configuration.direction, Perspective::Dual, None),
            None => hutspot::color::BLACK.into(),
        };
        add_line2(
            &mut gizmos_cache.raycaster,
            u,
            v,
            n * 0.01,
            color,
            Vec3::new(
                mesh_resmut.properties.translation.x as f32,
                mesh_resmut.properties.translation.y as f32,
                mesh_resmut.properties.translation.z as f32,
            ),
            mesh_resmut.properties.scale,
        );
    }

    if configuration.ui_is_hovered.iter().any(|&x| x) || cursor_ray.is_none() {
        return;
    }

    let intersections = raycast.cast_ray(cursor_ray.unwrap(), &default());
    if intersections.is_empty() || query.single() != intersections[0].0 {
        return;
    }

    if keyboard.pressed(KeyCode::ControlLeft) {
        return;
    }

    for ev in evr_motion.read() {
        if (ev.delta.x.abs() + ev.delta.y.abs()) > 2. {
            return;
        }
    }

    for ev in evr_scroll.read() {
        if (ev.x.abs() + ev.y.abs()) > 0. {
            return;
        }
    }

    let intersection = &intersections[0].1;
    if intersection.triangle().is_none() {
        return;
    }

    let position = hutspot::draw::invert_transform_coordinates(
        vec3_to_vector3d(intersection.position()),
        mesh_resmut.properties.translation,
        mesh_resmut.properties.scale,
    );
    let verts = [
        vec3_to_vector3d(intersection.triangle().unwrap()[0].into()),
        vec3_to_vector3d(intersection.triangle().unwrap()[1].into()),
        vec3_to_vector3d(intersection.triangle().unwrap()[2].into()),
    ]
    .into_iter()
    .map(|p| hutspot::draw::invert_transform_coordinates(p, mesh_resmut.properties.translation, mesh_resmut.properties.scale))
    .map(|p| mesh_resmut.vertex_lookup.nearest(&p.into()).1)
    .sorted_by(|&a, &b| {
        mesh_resmut
            .mesh
            .position(a)
            .metric_distance(&position)
            .partial_cmp(&mesh_resmut.mesh.position(b).metric_distance(&position))
            .unwrap()
    })
    .collect_vec();

    if mesh_resmut.mesh.face_with_verts(&verts).is_none() {
        return;
    }
    let face_id = mesh_resmut.mesh.face_with_verts(&verts).unwrap();
    let edgepair = mesh_resmut.mesh.edges_in_face_with_vert(face_id, verts[0]).unwrap();

    let u = mesh_resmut.mesh.midpoint(edgepair[0]);
    let v = mesh_resmut.mesh.midpoint(edgepair[1]);
    let n = mesh_resmut.mesh.normal(mesh_resmut.mesh.face(edgepair[0]));
    let color = to_color(configuration.direction, Perspective::Dual, None);
    add_line2(
        &mut gizmos_cache.raycaster,
        u,
        v,
        n * 0.01,
        color,
        Vec3::new(
            mesh_resmut.properties.translation.x as f32,
            mesh_resmut.properties.translation.y as f32,
            mesh_resmut.properties.translation.z as f32,
        ),
        mesh_resmut.properties.scale,
    );

    if keyboard.pressed(KeyCode::Space) {
        // Find closest loop.
        let option_a = [edgepair[0], edgepair[1]];
        let option_b = [edgepair[1], edgepair[0]];

        if let Some(loop_id) = solution.current_solution.loops.keys().find(|&loop_id| {
            let edges = solution.current_solution.get_pairs_of_loop(loop_id);
            edges.contains(&option_a) || edges.contains(&option_b)
        }) {
            if mouse.just_pressed(MouseButton::Left) || mouse.just_released(MouseButton::Left) {
                mouse.clear_just_pressed(MouseButton::Left);
                mouse.clear_just_released(MouseButton::Left);

                let mut new_sol = solution.current_solution.clone();
                new_sol.del_loop(loop_id);
                if new_sol.reconstruct_solution(configuration.unit_cubes).is_ok() {
                    solution.current_solution = new_sol;
                    solution.next[0].clear();
                    solution.next[1].clear();
                    solution.next[2].clear();
                    cache.cache[0].clear();
                    cache.cache[1].clear();
                    cache.cache[2].clear();
                }
                return;
            }
        }
        return;
    }

    // If shift button is pressed, we want to show the closest solution to the current face vertex combination.
    if keyboard.pressed(KeyCode::ShiftLeft) {
        keyboard.clear_just_pressed(KeyCode::ShiftLeft);

        // Find the closest face vertex combination to the current face vertex combination.
        // Map to distance. Then get the solution with the smallest distance.
        let closest_solution = solution.next[configuration.direction as usize]
            .iter()
            .map(|(&edgepair, sol)| {
                (
                    ((mesh_resmut.mesh.midpoint(edgepair[0]) + mesh_resmut.mesh.midpoint(edgepair[1])) / 2.).metric_distance(&position),
                    sol,
                    edgepair,
                )
            })
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        if let Some((_, closest_solution, signature)) = closest_solution {
            configuration.selected = Some(signature);

            if let Some(some_solution) = closest_solution {
                for loop_id in some_solution.loops.keys() {
                    let direction = some_solution.loop_to_direction(loop_id);
                    let color = to_color(direction, Perspective::Dual, None);
                    for &edgepair in &some_solution.get_pairs_of_loop(loop_id) {
                        let u = mesh_resmut.mesh.midpoint(edgepair[0]);
                        let v = mesh_resmut.mesh.midpoint(edgepair[1]);
                        let n = mesh_resmut.mesh.edge_normal(edgepair[0]);
                        add_line2(
                            &mut gizmos_cache.raycaster,
                            u,
                            v,
                            n * 0.05,
                            color,
                            Vec3::new(
                                mesh_resmut.properties.translation.x as f32,
                                mesh_resmut.properties.translation.y as f32,
                                mesh_resmut.properties.translation.z as f32,
                            ),
                            mesh_resmut.properties.scale,
                        );
                    }
                }

                // If the right mouse button is pressed, we want to save the candidate solution as the current solution.
                if mouse.just_pressed(MouseButton::Left) || mouse.just_released(MouseButton::Left) {
                    mouse.clear_just_pressed(MouseButton::Left);
                    mouse.clear_just_released(MouseButton::Left);

                    solution.current_solution = some_solution.clone();
                    solution.next[0].clear();
                    solution.next[1].clear();
                    solution.next[2].clear();
                    cache.cache[0].clear();
                    cache.cache[1].clear();
                    cache.cache[2].clear();
                    return;
                }
            }
        }

        return;
    }

    if !mouse.pressed(MouseButton::Left) {
        return;
    }

    if mouse.just_pressed(MouseButton::Left) || mouse.just_released(MouseButton::Left) {
        let selected_edges = mesh_resmut.mesh.edges_in_face_with_vert(face_id, verts[0]).unwrap();
        let measure = |a: f64| a.powi(10);

        if let Some((candidate_loop, _)) = solution.current_solution.construct_unbounded_loop(
            selected_edges,
            configuration.direction,
            &mesh_resmut.flow_graphs[configuration.direction as usize],
            measure,
        ) {
            let mut candidate_solution = solution.current_solution.clone();
            candidate_solution.add_loop(Loop {
                edges: candidate_loop,
                direction: configuration.direction,
            });

            candidate_solution.reconstruct_solution(configuration.unit_cubes);

            solution.next[configuration.direction as usize].insert([selected_edges[0], selected_edges[1]], Some(candidate_solution));
        } else {
            solution.next[configuration.direction as usize].insert([selected_edges[0], selected_edges[1]], None);
        }
    }

    // // The left mouse button is pressed, and no solution has been computed yet.
    // // We will compute a solution for this face and vertex combination.
    // let mut timer = hutspot::timer::Timer::new();
    // solution.next[configuration.direction as usize].insert(edgepair, None);

    // // Compute the occupied edges (edges that are already part of the solution, they are covered by loops)
    // let occupied = solution.current_solution.occupied_edgepairs();
    // timer.report("Computed `occupied`, hashmap of occupied edges");
    // timer.reset();

    // let filter = |(a, b): (&[EdgeID; 2], &[EdgeID; 2])| {
    //     !occupied.contains(a)
    //         && !occupied.contains(b)
    //         && [a[0], a[1], b[0], b[1]].iter().all(|&edge| {
    //             solution
    //                 .current_solution
    //                 .loops_on_edge(edge)
    //                 .iter()
    //                 .filter(|&&loop_id| solution.current_solution.loop_to_direction(loop_id) == configuration.direction)
    //                 .count()
    //                 == 0
    //         })
    // };

    // let g_original = &mesh_resmut.flow_graphs[configuration.direction as usize];
    // let g = g_original.filter(filter);

    // let measure = |(a, b, c): (f64, f64, f64)| configuration.alpha * a.powi(5) + (1. - configuration.alpha) * b.powi(5);

    // // Starting edges.
    // let [e1, e2] = mesh_resmut.mesh.edges_in_face_with_vert(face_id, verts[0]).unwrap();
    // let a = g.node_to_index(&[e1, e2]).unwrap();
    // let b = g.node_to_index(&[e2, e1]).unwrap();
    // let option_a = g
    //     .shortest_cycle(a, &measure)
    //     .unwrap_or_default()
    //     .into_iter()
    //     .flat_map(|node_index| g.index_to_node(node_index).unwrap().to_owned())
    //     .collect_vec();
    // let option_b = g
    //     .shortest_cycle(b, &measure)
    //     .unwrap_or_default()
    //     .into_iter()
    //     .flat_map(|node_index| g.index_to_node(node_index).unwrap().to_owned())
    //     .collect_vec();

    // // The path may contain self intersections. We can remove these.
    // // If duplicated vertices are present, remove everything between them.
    // let mut cleaned_option_a = vec![];
    // for edge_id in option_a {
    //     if cleaned_option_a.contains(&edge_id) {
    //         cleaned_option_a = cleaned_option_a.into_iter().take_while(|&x| x != edge_id).collect_vec();
    //     }
    //     cleaned_option_a.push(edge_id);
    // }

    // let mut cleaned_option_b = vec![];
    // for edge_id in option_b {
    //     if cleaned_option_b.contains(&edge_id) {
    //         cleaned_option_b = cleaned_option_b.into_iter().take_while(|&x| x != edge_id).collect_vec();
    //     }
    //     cleaned_option_b.push(edge_id);
    // }

    // if cleaned_option_a.len() <= 5 && cleaned_option_b.len() <= 5 {
    //     println!("Path is empty and/or invalid.");
    //     return;
    // }

    // if cleaned_option_a.len() <= 5 {
    //     cleaned_option_a = cleaned_option_b.clone();
    // }

    // let best_option = if cleaned_option_b.len() < cleaned_option_a.len() {
    //     cleaned_option_b
    // } else {
    //     cleaned_option_a
    // };

    // if best_option.len() <= 5 {
    //     return;
    // }

    // timer.report("Path computation");
    // timer.reset();

    // let mut real_solution = solution.current_solution.clone();

    // real_solution.add_loop(Loop {
    //     edges: best_option,
    //     direction: configuration.direction,
    // });
    // real_solution.reconstruct_solution(configuration.unit_cubes);

    // solution.next[configuration.direction as usize].insert(edgepair, Some(real_solution));
}
