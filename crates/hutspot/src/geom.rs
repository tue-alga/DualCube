use crate::consts::EPS;
use approx::AbsDiffEq;

pub type Vector2D = nalgebra::SVector<f64, 2>;
pub type Vector3D = nalgebra::SVector<f64, 3>;

/// Represents the orientation of three points in 3D space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    C,   // Collinear
    CW,  // Clockwise
    CCW, // Counterclockwise
}

/// Represents the type of intersection between line segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntersectionType {
    Proper,
    Endpoint,
}

/// Calculates the area of a triangle `t` in 3D space.
/// # Arguments
/// * `t` - A tuple of three vertices representing the triangle.
/// # Returns
/// * `f64` - The area of the triangle.
///
/// # Source
/// This method uses the cross product of vectors to find the area of a triangle in 3D space. For more details, see [Wikipedia](https://en.wikipedia.org/wiki/Area_of_a_triangle#Using_vector_cross_products).
#[must_use]
#[inline]
pub fn calculate_triangle_area(t: (Vector3D, Vector3D, Vector3D)) -> f64 {
    (t.1 - t.0).cross(&(t.2 - t.0)).magnitude() * 0.5
}

/// Checks if four points `a`, `b`, `c`, and `d` are coplanar.
/// # Arguments
/// * `a` - First point.
/// * `b` - Second point.
/// * `c` - Third point.
/// * `d` - Fourth point.
/// # Returns
/// * `bool` - `true` if the points are coplanar, `false` otherwise.
///
/// # Source
/// This method is based on the concept of coplanarity in vector mathematics, which can be determined using the scalar triple product. For more details, see [Wikipedia](https://en.wikipedia.org/wiki/Coplanarity).
#[must_use]
#[inline]
pub fn are_points_coplanar(a: Vector3D, b: Vector3D, c: Vector3D, d: Vector3D) -> bool {
    (b - a).cross(&(c - a)).dot(&(d - a)) == 0.
}

/// Calculates the orientation of three points `a`, `b`, `c`, with normal `n`, in 3D space.
/// # Arguments
/// * `a` - First point.
/// * `b` - Second point.
/// * `c` - Third point.
/// * `n` - Normal vector of the plane containing the points.
/// # Returns
/// * `Orientation` - The orientation of the points.
///
/// # Source
/// This method is based on the right-hand rule for the cross product and the dot product to determine the relative orientation of the points in a plane. For more details, see [Wikipedia](https://en.wikipedia.org/wiki/Orientation_(vector_space)).
#[must_use]
#[inline]
pub fn calculate_orientation(a: Vector3D, b: Vector3D, c: Vector3D, n: Vector3D) -> Orientation {
    let orientation = (b - a).cross(&(c - a)).dot(&n);
    if orientation > 0. {
        Orientation::CCW
    } else if orientation < 0. {
        Orientation::CW
    } else {
        Orientation::C
    }
}

/// Calculates the clockwise angle between `a->b` and `a->c` with the three points `a`, `b`, `c`, with normal `n`, in 3D space.
/// # Arguments
/// * `a` - First point.
/// * `b` - Second point.
/// * `c` - Third point.
/// * `n` - Normal vector of the plane containing the points.
/// # Returns
/// * `f64` - The clockwise angle between the two vectors.
///
/// # Source
/// This method uses the dot product to calculate the angle between two vectors in 3D space. For more details, see [Wikipedia](https://en.wikipedia.org/wiki/Dot_product).
///
/// # Example
/// ```
/// use hutspot::geom::calculate_clockwise_angle;
/// use hutspot::geom::calculate_orientation;
/// use hutspot::geom::Vector3D;
/// let a = Vector3D::new(0., 0., 0.);
/// let up = Vector3D::new(0., 1., 0.);
/// let down = Vector3D::new(0., -1., 0.);
/// let right = Vector3D::new(1., 0., 0.);
/// let left = Vector3D::new(-1., 0., 0.);
/// let n = Vector3D::new(0., 0., 1.);
/// let epsilon = 1e-6;
/// let tests = vec![
///     (a, up, right, n, std::f64::consts::FRAC_PI_2), // 90 degrees
///     (a, right, up, n, 3. * std::f64::consts::FRAC_PI_2),  // 270 degrees
///     (a, down, right, n, 3. * std::f64::consts::FRAC_PI_2),  // 270 degrees
///     (a, right, down, n, std::f64::consts::FRAC_PI_2),  // 90 degrees
///     (a, up, left, n, 3. * std::f64::consts::FRAC_PI_2),  // 270 degrees
///     (a, left, up, n, std::f64::consts::FRAC_PI_2),  // 90 degrees
///     (a, down, left, n, std::f64::consts::FRAC_PI_2),  // 90 degrees
///     (a, left, down, n, 3. * std::f64::consts::FRAC_PI_2),  // 270 degrees
///     (a, up, up, n, 0.0),  // 0 degrees
///     (a, right, right, n, 0.0),  // 0 degrees
///     (a, down, down, n, 0.0),  // 0 degrees
///     (a, left, left, n, 0.0),  // 0 degrees
///     (a, up, down, n, std::f64::consts::PI),  // 180 degrees
///     (a, right, left, n, std::f64::consts::PI),  // 180 degrees
///     (a, down, up, n, std::f64::consts::PI),  // 180 degrees
///     (a, left, right, n, std::f64::consts::PI),  // 180 degrees
/// ];
/// for (a, b, c, n, expected) in tests {
///     let angle = calculate_clockwise_angle(a, b, c, n);
///     assert!((angle - expected).abs() < epsilon, "calculate_clockwise_angle({a:?}, {b:?}, {c:?}, {n:?}) = {angle}, but should be: {expected}");
/// }
/// ```
#[must_use]
#[inline]
pub fn calculate_clockwise_angle(a: Vector3D, b: Vector3D, c: Vector3D, n: Vector3D) -> f64 {
    let ab = (b - a).normalize();
    let ac = (c - a).normalize();
    let angle = ab.angle(&ac);
    if calculate_orientation(a, b, c, n) == Orientation::CCW {
        2. * std::f64::consts::PI - angle
    } else {
        angle
    }
}

/// Projects point `point` onto a plane `plane` along reference `reference`.
/// # Arguments
/// * `point` - The point to project.
/// * `plane` - A tuple representing the plane (two vectors).
/// * `reference` - A reference point on the plane.
/// # Returns
/// * `Vector2D` - The projected point in 2D space.
#[must_use]
#[inline]
pub fn project_point_onto_plane(point: Vector3D, plane: (Vector3D, Vector3D), reference: Vector3D) -> Vector2D {
    Vector2D::new((point - reference).dot(&plane.0), (point - reference).dot(&plane.1))
}

/// Checks if point `p` is inside the triangle `t` using barycentric coordinates.
/// # Arguments
/// * `p` - The point to check.
/// * `t` - A tuple of three vertices representing the triangle.
/// # Returns
/// * `bool` - `true` if the point is inside the triangle, `false` otherwise.
///
/// # Source
/// This method is based on the use of barycentric coordinates to determine if a point lies within a triangle. For more details, see [Wikipedia](https://en.wikipedia.org/wiki/Barycentric_coordinate_system).
///
/// # Example
/// ```
/// use hutspot::geom::is_point_inside_triangle;
/// use hutspot::geom::Vector3D;
/// use hutspot::consts::EPS;
/// let triangle = (Vector3D::new(0., 0., 0.), Vector3D::new(1., 0., 0.), Vector3D::new(0., 1., 0.));
/// let point = Vector3D::new(0.5, 0.5, 0.0);
/// let epsilon_x = Vector3D::new(EPS, 0.0, 0.0);
/// let epsilon_y = Vector3D::new(0.0, EPS, 0.0);
/// let epsilon_z = Vector3D::new(0.0, 0.0, EPS);
/// let tests = vec![
///     (point, triangle, true),              // On the boundary
///     (point + epsilon_x, triangle, false), // Outside the triangle (by epsilon)
///     (point - epsilon_x, triangle, true),  // Inside the triangle (by epsilon)
///     (point + epsilon_y, triangle, false), // Outside the triangle (by epsilon)
///     (point - epsilon_y, triangle, true),  // Inside the triangle (by epsilon)
///     (point + epsilon_z, triangle, false), // Outside the triangle (by epsilon in z axis)
///     (point - epsilon_z, triangle, false), // Outside the triangle (by epsilon in z axis)
/// ];
///
/// for (point, triangle, expected) in tests {
///     let inside = is_point_inside_triangle(point, triangle);
///     assert_eq!(inside, expected, "inside_triangle({point:?}, {triangle:?}) = {inside:?}, but should be: {expected:?}");
/// }
/// ```
#[must_use]
#[inline]
pub fn is_point_inside_triangle(p: Vector3D, t: (Vector3D, Vector3D, Vector3D)) -> bool {
    let s1 = calculate_triangle_area((t.0, t.1, p));
    let s2 = calculate_triangle_area((t.1, t.2, p));
    let s3 = calculate_triangle_area((t.2, t.0, p));
    let st = calculate_triangle_area(t);
    (s1 + s2 + s3 - st).abs() < EPS && (0.0 - EPS..=st + EPS).contains(&s1) && (0.0 - EPS..=st + EPS).contains(&s2) && (0.0 - EPS..=st + EPS).contains(&s3)
}

/// Checks whether the element `a` lies within the range `(b..=c)` or `(c..=b)`.
/// # Arguments
/// * `a` - The element to check.
/// * `b` - The first boundary value.
/// * `c` - The second boundary value.
/// # Returns
/// * `bool` - `true` if `a` lies within the specified ranges, `false` otherwise.
///
/// # Example
/// ```
/// use hutspot::geom::is_within_inclusive_range;
/// let tests = vec![
///     (0.5, 0.0, 1.0, true),  // Inside the range
///     (0.0, 0.0, 1.0, true),  // At the lower boundary
///     (1.0, 0.0, 1.0, true),  // At the upper boundary
///     (0.0, 1.0, 0.0, true),  // Inside the range (reversed)
///     (1.0, 1.0, 0.0, true),  // Inside the range (reversed)
///     (0.0, 1.0, 0.0, true),  // At the lower boundary (reversed)
///     (1.0, 0.0, 1.0, true),  // At the upper boundary (reversed)
///     (0.0, 0.0, 0.0, true),  // At the same point
/// ];
/// for (a, b, c, expected) in tests {
///     let result = is_within_inclusive_range(a, b, c);
///     assert_eq!(result, expected, "is_within_inclusive_range({a}, {b}, {c}) = {result}, but should be: {expected}");
/// }
/// ```
#[must_use]
#[inline]
pub fn is_within_inclusive_range(a: f64, b: f64, c: f64) -> bool {
    if b < c {
        (b..=c).contains(&a)
    } else {
        (c..=b).contains(&a)
    }
}

/// Calculates the intersection of two line segments (`p_u`, `p_v`) and (`q_u`, `q_v`) in 2D space.
/// # Arguments
/// * `p_u` - First point of the first line segment.
/// * `p_v` - Second point of the first line segment.
/// * `q_u` - First point of the second line segment.
/// * `q_v` - Second point of the second line segment.
/// # Returns
/// * `Option<Intersection2D>` - The intersection point and type, or `None` if no intersection.
///
/// # Source
/// This method uses the parametric form of the line equation to calculate the intersection point of two line segments in 2D space. For more details, see [Wikipedia](https://en.wikipedia.org/wiki/Intersection_(geometry)#Two_line_segments).
///
/// # Example
/// ```
/// use hutspot::geom::calculate_2d_lineseg_intersection;
/// use hutspot::geom::Vector2D;
/// use hutspot::geom::IntersectionType::{Endpoint, Proper};
/// use hutspot::consts::EPS;
/// use approx::AbsDiffEq;
/// let p1 = Vector2D::new(0., 0.);
/// let p2 = Vector2D::new(1., 1.);
/// let mid = Vector2D::new(0.5, 0.5);
/// let p3 = Vector2D::new(1., 0.);
/// let p4 = Vector2D::new(0., 1.);
/// let epsilon_x = Vector2D::new(EPS, 0.0);
/// let epsilon_y = Vector2D::new(0.0, EPS);
/// let tests = vec![
///     (p1, p2, p3, p4, Some((mid, Proper))),
///     (p2, p1, p4, p3, Some((mid, Proper))),
///     (p1, p3, p2, p4, None),
///     (p3, p1, p4, p2, None),
///     (p1, p3, p2, p3, Some((p3, Endpoint))),
///     (p1, p3, p2, p3 + epsilon_y, None),
///     (p1, p3, p2, p3 - epsilon_y, Some((p3, Endpoint))),
///     (p1, p3, p2, p3 + epsilon_x, None),
///     (p1, p3, p2, p3 - epsilon_x, Some((p3 - epsilon_x, Proper))),
///     (p1, p3 + epsilon_y, p2, p3, Some((p3 + epsilon_y, Proper))),
///     (p1, p3 - epsilon_y, p2, p3, None),
///     (p1, p3 + epsilon_x, p2, p3, Some((p3, Endpoint))),
///     (p1, p3 - epsilon_x, p2, p3, None),
/// ];
///
/// for (a, b, c, d, expected) in tests {
///     let intersection = calculate_2d_lineseg_intersection(a, b, c, d);
///     assert!(
///         match (intersection, expected) {
///             (Some((a, _)), Some((b, _))) => a.abs_diff_eq(&b, EPS),
///             (None, None) => true,
///             _ => false,
///         },
///         "intersection({a:?}, {b:?}, {c:?}, {d:?}) = {intersection:?}, but should be: {expected:?}"
///     );
/// }
/// ```
#[must_use]
pub fn calculate_2d_lineseg_intersection(p_u: Vector2D, p_v: Vector2D, q_u: Vector2D, q_v: Vector2D) -> Option<(Vector2D, IntersectionType)> {
    let (x1, x2, x3, x4, y1, y2, y3, y4) = (p_u.x, p_v.x, q_u.x, q_v.x, p_u.y, p_v.y, q_u.y, q_v.y);

    let t_numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4);
    let u_numerator = (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2);
    let denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

    if denominator.abs() < EPS {
        return None;
    }

    if is_within_inclusive_range(t_numerator, 0.0, denominator) {
        let t = t_numerator / denominator;
        if t.abs() < EPS {
            return Some((p_u, IntersectionType::Endpoint));
        }
        if (t - 1.0).abs() < EPS {
            return Some((p_v, IntersectionType::Endpoint));
        }
        let sx_t = t.mul_add(x2 - x1, x1);
        let sy_t = t.mul_add(y2 - y1, y1);
        let s_t = Vector2D::new(sx_t, sy_t);

        Some((s_t, IntersectionType::Proper))
    } else if is_within_inclusive_range(u_numerator, 0.0, denominator) {
        let u = u_numerator / denominator;
        if u.abs() < EPS {
            return Some((q_u, IntersectionType::Endpoint));
        }
        if (u - 1.0).abs() < EPS {
            return Some((q_v, IntersectionType::Endpoint));
        }
        let sx_u = u.mul_add(x4 - x3, x3);
        let sy_u = u.mul_add(y4 - y3, y3);
        let s_u = Vector2D::new(sx_u, sy_u);

        Some((s_u, IntersectionType::Proper))
    } else {
        None
    }
}

/// Calculates the intersection of two line segments (`p_u`, `p_v`) and (`q_u`, `q_v`) in 3D space.
/// # Arguments
/// * `p_u` - First point of the first line segment.
/// * `p_v` - Second point of the first line segment.
/// * `q_u` - First point of the second line segment.
/// * `q_v` - Second point of the second line segment.
/// # Returns
/// * `Option<Intersection3D>` - The intersection point and type, or `None` if no intersection.
///
/// # Example
/// ```
/// use hutspot::geom::calculate_3d_lineseg_intersection;
/// use hutspot::geom::Vector3D;
/// use hutspot::geom::IntersectionType::{Endpoint, Proper};
/// use hutspot::consts::EPS;
/// use approx::AbsDiffEq;
/// let p1 = Vector3D::new(0.0, 0.0, 0.0);
/// let p2 = Vector3D::new(1.0, 1.0, 0.0);
/// let mid = Vector3D::new(0.5, 0.5, 0.0);
/// let p3 = Vector3D::new(1.0, 0.0, 0.0);
/// let p4 = Vector3D::new(0.0, 1.0, 0.0);
/// let d1 = Vector3D::new(0.0, 0.0, 1.0);
/// let d2 = Vector3D::new(0.0, 0.0, -1.0);
/// let d3 = Vector3D::new(424242., 424242., 424242.0);
/// let d4 = Vector3D::new(-424242., -424242., -424242.0);
/// let epsilon_x = Vector3D::new(EPS, 0.0, 0.0);
/// let epsilon_y = Vector3D::new(0.0, EPS, 0.0);
/// let epsilon_z = Vector3D::new(0.0, 0.0, EPS);
/// let tests = vec![
///     (p1, p2, p3, p4, Some((mid, Proper))),
///     (p2, p1, p4, p3, Some((mid, Proper))),
///     (p1 + d1, p2 + d1, p3 + d1, p4 + d1, Some((mid + d1, Proper))),
///     (p1 + d2, p2 + d2, p3 + d2, p4 + d2, Some((mid + d2, Proper))),
///     (p1 + d3, p2 + d3, p3 + d3, p4 + d3, Some((mid + d3, Proper))),
///     (p1 + d4, p2 + d4, p3 + d4, p4 + d4, Some((mid + d4, Proper))),
///     (p1, p3, p2, p4, None),
///     (p1, p3, p2, p3 - epsilon_y, Some((p3, Endpoint))),
///     (p1, p3, p2, p3 + epsilon_x, None),
///     (p1, p3, p2, p3 - epsilon_x, Some((p3 - epsilon_x, Proper))),
///     (p1, p3 + epsilon_y, p2, p3, Some((p3 + epsilon_y, Proper))),
///     (p1, p3 - epsilon_y, p2, p3, None),
///     (p1, p3 + epsilon_x, p2, p3, Some((p3, Endpoint))),
///     (p1, p3 - epsilon_x, p2, p3, None),
///     (p1, p3 + epsilon_z, p2, p3, None),
///     (p1, p3 - epsilon_z, p2, p3, None),
///     (p1, p3, p2, p3 + epsilon_z, None),
///     (p1, p3, p2, p3 - epsilon_z, None),
/// ];
///
/// for (a, b, c, d, expected) in tests {
///     let intersection = calculate_3d_lineseg_intersection(a, b, c, d);
///     assert!(
///         match (intersection, expected) {
///             (Some((a, _)), Some((b, _))) => a.abs_diff_eq(&b, EPS),
///             (None, None) => true,
///             _ => false,
///         },
///         "intersection({a:?}, {b:?}, {c:?}, {d:?}) = {intersection:?}, but should be: {expected:?}"
///     );
/// }
/// ```
#[must_use]
pub fn calculate_3d_lineseg_intersection(p_u: Vector3D, p_v: Vector3D, q_u: Vector3D, q_v: Vector3D) -> Option<(Vector3D, IntersectionType)> {
    if !are_points_coplanar(p_u, p_v, q_u, q_v) {
        return None;
    }

    let p = p_v - p_u;
    let q = q_v - q_u;
    let normal_vector = p.cross(&q).normalize();
    let reference_point = p_u;
    let plane = (p.normalize(), p.cross(&normal_vector).normalize());

    calculate_2d_lineseg_intersection(
        project_point_onto_plane(p_u, plane, reference_point),
        project_point_onto_plane(p_v, plane, reference_point),
        project_point_onto_plane(q_u, plane, reference_point),
        project_point_onto_plane(q_v, plane, reference_point),
    )
    .map(|(point_in_2d, intersection_type)| {
        let point_in_3d = reference_point + (plane.0 * point_in_2d.x) + (plane.1 * point_in_2d.y);
        (point_in_3d, intersection_type)
    })
}
