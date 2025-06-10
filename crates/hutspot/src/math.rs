use itertools::Itertools;

/// Calculates the average of a list of elements.
/// # Arguments
/// * `list` - An iterator over elements to average.
/// # Returns
/// * `T` - The average value.
///
/// # Example
/// ```
/// use hutspot::math::calculate_average_f32;
/// let list = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let average = calculate_average_f32(list.into_iter());
/// assert_eq!(average, 3.0);
/// ```
#[must_use]
#[inline]
pub fn calculate_average_f32<T>(list: impl Iterator<Item = T>) -> T
where
    T: Default + std::ops::Add<Output = T> + std::ops::Div<f32, Output = T> + std::iter::Sum<T>,
{
    let (sum, count) = list.fold((T::default(), 0.0), |(sum, count), elem| {
        (sum + elem, count + 1.0)
    });
    sum / count
}

/// Calculates the average of a list of elements.
/// # Arguments
/// * `list` - An iterator over elements to average.
/// # Returns
/// * `T` - The average value.
///
/// # Example
/// ```
/// use hutspot::math::calculate_average_f64;
/// let list = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let average = calculate_average_f64(list.into_iter());
/// assert_eq!(average, 3.0);
///
/// use hutspot::geom::Vector2D;
/// let list = vec![Vector2D::new(1.0, 2.0), Vector2D::new(3.0, 4.0), Vector2D::new(5.0, 6.0)];
/// let average = calculate_average_f64(list.into_iter());
/// assert_eq!(average, Vector2D::new(3.0, 4.0));
/// ```
#[must_use]
#[inline]
pub fn calculate_average_f64<T>(list: impl Iterator<Item = T>) -> T
where
    T: Default + std::ops::Add<Output = T> + std::ops::Div<f64, Output = T> + std::iter::Sum<T>,
{
    let (sum, count) = list.fold((T::default(), 0.0), |(sum, count), elem| {
        (sum + elem, count + 1.0)
    });
    sum / count
}

// #[must_use]
// pub fn convert_3d_to_2d(point: Vector3D, reference: Vector3D) -> Vector2D {
//     let alpha = point.angle_between(reference);
//     Vector2D::new(point.length() * alpha.cos(), point.length() * alpha.sin())
// }

// // Draw objects by returning a list of lines to render
// #[must_use]
// pub fn draw_vertex(p: Vector3D, n: Vector3D) -> Vec<(Vector3D, Vector3D)> {
//     vec![(p, p + n)]
// }

// #[must_use]
// pub fn draw_line(p1: Vector3D, p2: Vector3D) -> Vec<(Vector3D, Vector3D)> {
//     vec![(p1, p2)]
// }

// #[must_use]
// pub fn draw_triangle(p1: Vector3D, p2: Vector3D, p3: Vector3D) -> Vec<(Vector3D, Vector3D)> {
//     vec![(p1, p2), (p2, p3), (p3, p1)]
// }

// #[must_use]
// pub fn draw_quad(
//     p1: Vector3D,
//     p2: Vector3D,
//     p3: Vector3D,
//     p4: Vector3D,
// ) -> Vec<(Vector3D, Vector3D)> {
//     vec![(p1, p2), (p2, p3), (p3, p4), (p4, p1)]
// }

// #[must_use]
// pub fn draw_polygon(vertices: &[Vector3D]) -> Vec<(Vector3D, Vector3D)> {
//     let mut lines = Vec::new();
//     for i in 0..vertices.len() {
//         lines.push((vertices[i], vertices[(i + 1) % vertices.len()]));
//     }
//     lines
// }

#[must_use]
pub fn intersection_in_sequence(elem_a: usize, elem_b: usize, sequence: &[usize]) -> bool {
    let mut sequence_copy = sequence.to_owned();
    sequence_copy.retain(|&elem| elem == elem_a || elem == elem_b);
    debug_assert!(sequence_copy.len() == 4, "{sequence_copy:?}");
    sequence_copy.dedup();
    sequence_copy.len() >= 4
}

#[must_use]
pub fn set_intersection<T: std::cmp::PartialEq + Clone>(
    collection_a: &[T],
    collection_b: &[T],
) -> Vec<T> {
    let mut intesection = collection_b.to_owned();
    intesection.retain(|edge_id| collection_a.contains(edge_id));
    intesection
}

#[must_use]
pub fn wrap_pairs<T: Copy>(sequence: &[T]) -> Vec<(T, T)> {
    sequence
        .iter()
        .cycle()
        .copied()
        .take(sequence.len() + 1)
        .tuple_windows()
        .collect()
}
