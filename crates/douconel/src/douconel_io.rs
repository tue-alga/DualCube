use crate::{
    douconel::{Douconel, MeshError},
    douconel_embedded::{EmbeddedMeshError, HasPosition},
};
use bimap::BiHashMap;
use hutspot::geom::Vector3D;
use itertools::Itertools;
use slotmap::Key;
use std::io::Write;
use std::{
    fs::OpenOptions,
    io::{BufRead, BufReader},
    path::PathBuf,
};

impl<VertID: Key, V: Default + HasPosition, EdgeID: Key, E: Default, FaceID: Key, F: Default> Douconel<VertID, V, EdgeID, E, FaceID, F> {
    pub fn obj_to_elements(reader: impl BufRead) -> Result<(Vec<Vector3D>, Vec<Vec<usize>>), obj::ObjError> {
        let obj = obj::ObjData::load_buf(reader)?;
        let verts = obj.position.iter().map(|v| Vector3D::new(v[0].into(), v[1].into(), v[2].into())).collect_vec();
        let faces = obj.objects[0].groups[0]
            .polys
            .iter()
            .map(|f| f.0.iter().map(|v| v.0).collect_vec())
            .collect_vec();
        Ok((verts, faces))
    }

    pub fn stl_to_elements(mut reader: impl BufRead + std::io::Seek) -> Result<(Vec<Vector3D>, Vec<Vec<usize>>), std::io::Error> {
        let stl = stl_io::read_stl(&mut reader)?;
        let verts = stl.vertices.iter().map(|v| Vector3D::new(v[0].into(), v[1].into(), v[2].into())).collect_vec();
        let faces = stl.faces.iter().map(|f| f.vertices.to_vec()).collect_vec();
        Ok((verts, faces))
    }

    pub fn from_file(path: &PathBuf) -> Result<(Self, BiHashMap<usize, VertID>, BiHashMap<usize, FaceID>), EmbeddedMeshError<VertID, FaceID>> {
        match OpenOptions::new().read(true).open(path) {
            Ok(file) => match path.extension().unwrap().to_str() {
                Some("obj") => match Self::obj_to_elements(BufReader::new(file)) {
                    Ok((verts, faces)) => Self::from_embedded_faces(&faces, &verts),
                    Err(e) => Err(EmbeddedMeshError::MeshError(MeshError::Unknown(format!(
                        "Something went wrong while reading the OBJ file: {path:?}\nErr: {e}"
                    )))),
                },
                Some("stl") => match Self::stl_to_elements(BufReader::new(file)) {
                    Ok((verts, faces)) => Self::from_embedded_faces(&faces, &verts),
                    Err(e) => Err(EmbeddedMeshError::MeshError(MeshError::Unknown(format!(
                        "Something went wrong while reading the STL file: {path:?}\nErr: {e}"
                    )))),
                },
                _ => Err(EmbeddedMeshError::MeshError(MeshError::Unknown(format!("Unknown file extension: {path:?}",)))),
            },
            Err(e) => Err(EmbeddedMeshError::MeshError(MeshError::Unknown(format!(
                "Cannot read file: {path:?}\nErr: {e}"
            )))),
        }
    }

    pub fn write_to_obj(&self, path: &PathBuf) -> Result<(), std::io::Error> {
        let mut file = std::fs::File::create(path)?;

        let vert_id_to_int = self
            .vert_ids()
            .into_iter()
            .enumerate()
            .map(|(i, vert_id)| (vert_id, i + 1))
            .collect::<BiHashMap<VertID, usize>>();

        writeln!(
            file,
            "{}",
            self.vert_ids()
                .into_iter()
                .map(|vert_id| format!(
                    "v {x:.6} {y:.6} {z:.6}",
                    x = self.position(vert_id).x,
                    y = self.position(vert_id).y,
                    z = self.position(vert_id).z
                ))
                .join("\n")
        )?;

        writeln!(
            file,
            "{}",
            self.face_ids()
                .into_iter()
                .map(|face_id| {
                    format!(
                        "vn {x:.6} {y:.6} {z:.6}",
                        x = self.normal(face_id).x,
                        y = self.normal(face_id).y,
                        z = self.normal(face_id).z
                    )
                })
                .join("\n")
        )?;

        writeln!(
            file,
            "{}",
            self.face_ids()
                .into_iter()
                .map(|face_id| {
                    format!(
                        "f {}",
                        self.corners(face_id)
                            .iter()
                            .map(|vert_id| format!("{}", vert_id_to_int.get_by_left(vert_id).unwrap()))
                            .join(" ")
                    )
                })
                .join("\n")
        )?;

        Ok(())
    }
}
