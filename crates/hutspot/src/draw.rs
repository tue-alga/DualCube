use crate::geom::Vector3D;
use glam::Vec3;

// (p * s) + t = p'
#[must_use]
pub fn transform_coordinates(position: Vector3D, translation: Vector3D, scale: f32) -> Vector3D {
    position * scale as f64 + translation
}

// (p' - t) / s = p
#[must_use]
pub fn invert_transform_coordinates(
    position: Vector3D,
    translation: Vector3D,
    scale: f32,
) -> Vector3D {
    (position - translation) / scale as f64
}

pub struct DrawableLine {
    pub u: Vec3,
    pub v: Vec3,
}

impl DrawableLine {
    pub fn new(u: Vector3D, v: Vector3D) -> Self {
        Self {
            u: Vec3::new(u.x as f32, u.y as f32, u.z as f32),
            v: Vec3::new(v.x as f32, v.y as f32, v.z as f32),
        }
    }

    pub fn from_line(
        u: Vector3D,
        v: Vector3D,
        offset: Vector3D,
        translation: Vector3D,
        scale: f32,
    ) -> Self {
        Self::new(
            transform_coordinates(u, translation, scale) + offset,
            transform_coordinates(v, translation, scale) + offset,
        )
    }

    pub fn from_vertex(
        p: Vector3D,
        n: Vector3D,
        length: f32,
        translation: Vector3D,
        scale: f32,
    ) -> Self {
        Self::from_line(
            p,
            p + n * length as f64,
            Vector3D::new(0., 0., 0.),
            translation,
            scale,
        )
    }

    pub fn from_arrow(
        u: Vector3D,
        v: Vector3D,
        n: Vector3D,
        length: f32,
        offset: Vector3D,
        translation: Vector3D,
        scale: f32,
    ) -> [Self; 3] {
        let forward = (v - u) * length as f64;

        let cross = forward.cross(&n).normalize() * forward.magnitude();

        // height of wing
        const W1: f64 = 0.3;
        // width of wing
        const W2: f64 = 0.1;

        //wings
        let wing1 = W1 * -forward + W2 * cross;
        let wing2 = W1 * -forward - W2 * cross;

        [
            Self::from_line(u, u + forward, offset, translation, scale),
            Self::from_line(u + forward, u + forward + wing1, offset, translation, scale),
            Self::from_line(u + forward, u + forward + wing2, offset, translation, scale),
        ]
    }
}
