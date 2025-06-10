use itertools::Itertools;
use ordered_float::OrderedFloat;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

//
// 1 -> 2 <- 4 -> 6
//  \   |    ^
//   \  v    |
//    > 3 -> 5
//

/// Finds the shortest path from element `a` to element `b` using Dijkstra's algorithm.
///
/// # Arguments
/// * `a` - The starting element.
/// * `b` - The target element.
/// * `neighbor_function` - A function that returns the neighbors of a given element.
/// * `weight_function` - A function that returns the weight of the edge between two elements.
/// * `cache` - A mutable reference to a cache for storing the neighbors and weights for each element.
///
/// # Returns
/// * `Option<(Vec<T>, OrderedFloat<f64>)>` - An optional tuple containing the shortest path as a vector of elements
///    and the total weight of the path as an `OrderedFloat<f64>`. Returns `None` if no path is found.
///
/// # Example
/// ```
/// use hutspot::graph::find_shortest_path;
/// use ordered_float::OrderedFloat;
/// use std::collections::HashMap;
///
/// let neighbor_function = |node: u32| -> Vec<u32> {
///     match node {
///         1 => vec![2, 3],
///         2 => vec![3],
///         3 => vec![5],
///         4 => vec![2, 6],
///         5 => vec![4],
///         6 => vec![],
///         _ => vec![],
///     }
/// };
///
/// let weight_function = |a: u32, b: u32| -> OrderedFloat<f64> {
///     match (a, b) {
///         (1, 2) => 4.0.into(),
///         (1, 3) => 2.0.into(),
///         (2, 3) => 5.0.into(),
///         (3, 5) => 3.0.into(),
///         (4, 2) => 10.0.into(),
///         (4, 6) => 11.0.into(),
///         (5, 4) => 4.0.into(),
///         _ => OrderedFloat(f64::INFINITY),
///     }
/// };
///
/// let mut cache: HashMap<u32, Vec<(u32, OrderedFloat<f64>)>> = HashMap::new();
/// let result = find_shortest_path(1, 6, neighbor_function, weight_function, &mut cache);
/// assert!(result.is_some());
/// let (path, cost) = result.unwrap();
/// assert_eq!(path, vec![1, 3, 5, 4, 6]);
/// assert_eq!(cost, OrderedFloat(2.0 + 3.0 + 4.0 + 11.0));
///
/// let result = find_shortest_path(6, 1, neighbor_function, weight_function, &mut cache);
/// assert!(result.is_none());
/// ```
pub fn find_shortest_path<T: Eq + Hash + Clone + Copy>(
    a: T,
    b: T,
    neighbor_function: impl Fn(T) -> Vec<T>,
    weight_function: impl Fn(T, T) -> OrderedFloat<f64>,
) -> Option<(Vec<T>, OrderedFloat<f64>)> {
    pathfinding::prelude::dijkstra(
        &a,
        |&elem| {
            let neighbors = neighbor_function(elem)
                .iter()
                .map(|&neighbor| (neighbor, weight_function(elem, neighbor)))
                .collect_vec();
            neighbors
        },
        |&elem| elem == b,
    )
}

pub fn find_shortest_path_astar<T: Eq + Hash + Clone + Copy>(
    a: T,
    b: T,
    neighbor_function: impl Fn(T) -> Vec<T>,
    weight_function: impl Fn(T, T) -> OrderedFloat<f64>,
    heuristic_function: impl Fn(T, T) -> OrderedFloat<f64>,
) -> Option<(Vec<T>, OrderedFloat<f64>)> {
    pathfinding::directed::astar::astar(
        &a,
        |&elem| {
            let neighbors = neighbor_function(elem)
                .iter()
                .map(|&neighbor| (neighbor, weight_function(elem, neighbor)))
                .collect_vec();
            neighbors
        },
        |&elem| heuristic_function(elem, b),
        |&elem| elem == b,
    )
}

/// Finds the shortest cycle through element `a` using the `find_shortest_path` function (Dijkstra's algorithm).
///
/// # Arguments
/// * `a` - The starting element, which is also the element through which the cycle must pass.
/// * `neighbor_function` - A function that returns the neighbors of a given element.
/// * `weight_function` - A function that returns the weight of the edge between two elements.
/// * `cache` - A mutable reference to a cache for storing the neighbors and weights for each element.
///
/// # Returns
/// * `Option<(Vec<T>, OrderedFloat<f64>)>` - An optional tuple containing the shortest cycle as a vector of elements
///    and the total weight of the cycle as an `OrderedFloat<f64>`. Returns `None` if no cycle is found.
///
/// # Example
/// ```
/// use hutspot::graph::find_shortest_cycle;
/// use ordered_float::OrderedFloat;
/// use std::collections::HashMap;
///
/// let neighbor_function = |node: u32| -> Vec<u32> {
///     match node {
///         1 => vec![2, 3],
///         2 => vec![3],
///         3 => vec![5],
///         4 => vec![2, 6],
///         5 => vec![4],
///         6 => vec![],
///         _ => vec![],
///     }
/// };
///
/// let weight_function = |a: u32, b: u32| -> OrderedFloat<f64> {
///     match (a, b) {
///         (1, 2) => 4.0.into(),
///         (1, 3) => 2.0.into(),
///         (2, 3) => 5.0.into(),
///         (3, 5) => 3.0.into(),
///         (4, 2) => 10.0.into(),
///         (4, 6) => 11.0.into(),
///         (5, 4) => 4.0.into(),
///         _ => OrderedFloat(f64::INFINITY),
///     }
/// };
///
/// let mut cache: HashMap<u32, Vec<(u32, OrderedFloat<f64>)>> = HashMap::new();
/// let result = find_shortest_cycle(1, neighbor_function, weight_function, &mut cache);
/// assert!(result.is_none());
///
/// let result = find_shortest_cycle(3, neighbor_function, weight_function, &mut cache);
/// assert!(result.is_some());
/// let (path, cost) = result.unwrap();
/// assert_eq!(path, vec![3, 5, 4, 2]);
/// assert_eq!(cost, OrderedFloat(3.0 + 4.0 + 10.0 + 5.0));
/// ```
pub fn find_shortest_cycle<T: Eq + Hash + Clone + Copy>(
    a: T,
    neighbor_function: impl Fn(T) -> Vec<T>,
    weight_function: impl Fn(T, T) -> OrderedFloat<f64>,
) -> Option<(Vec<T>, OrderedFloat<f64>)> {
    neighbor_function(a)
        .iter()
        .filter_map(|&neighbor| find_shortest_path(neighbor, a, &neighbor_function, &weight_function))
        .sorted_by(|(_, cost1), (_, cost2)| cost1.cmp(cost2))
        .next()
        .map(|(path, score)| {
            let (last, rest) = path.split_last().unwrap();
            ([&[*last], rest].concat(), score + weight_function(a, *path.first().unwrap()))
        })
}

/// Finds the connected components of a graph.
///
/// # Example
/// ```
/// use hutspot::graph::find_ccs;
/// use std::collections::HashSet;
/// let neighbor_function_undirected = |node: u32| -> Vec<u32> {
///     match node {
///         1 => vec![2, 3],
///         2 => vec![1, 3, 4],
///         3 => vec![1, 2, 5],
///         4 => vec![2, 5, 6],
///         5 => vec![3, 4],
///         6 => vec![5],
///         _ => vec![],
///     }
/// };
///
/// let ccs = find_ccs(&vec![1, 2, 3, 4, 5, 6, 7], neighbor_function_undirected);
/// assert_eq!(ccs[0], HashSet::from([1, 2, 3, 4, 5, 6]));
/// assert_eq!(ccs[1], HashSet::from([7]));
/// ```
///
pub fn find_ccs<T>(nodes: &[T], neighbor_function: impl Fn(T) -> Vec<T>) -> Vec<HashSet<T>>
where
    T: Eq + Hash + Clone + Copy,
{
    let mut visited = HashSet::new();
    let mut ccs = vec![];
    for &node in nodes {
        if visited.contains(&node) {
            continue;
        }
        let cc = find_cc(node, &neighbor_function);
        visited.extend(cc.clone());
        ccs.push(cc);
    }
    ccs.into_iter().collect()
}

/// Finds the connected component of a graph that contains a specific node (or reachability from this specific node).
///
/// # Example
/// ```
/// use hutspot::graph::find_cc;
/// use std::collections::HashSet;
/// let neighbor_function = |node: u32| -> Vec<u32> {
///     match node {
///         1 => vec![2, 3],
///         2 => vec![3],
///         3 => vec![5],
///         4 => vec![2, 6],
///         5 => vec![4],
///         6 => vec![],
///         _ => vec![],
///     }
/// };
///
/// let cc = find_cc(1, neighbor_function);
/// assert_eq!(cc, HashSet::from([1, 2, 3, 4, 5, 6]));
///
/// let cc = find_cc(2, neighbor_function);
/// assert_eq!(cc, HashSet::from([2, 3, 5, 4, 6]));
///
/// let cc = find_cc(3, neighbor_function);
/// assert_eq!(cc, HashSet::from([2, 3, 5, 4, 6]));
///
/// let cc = find_cc(4, neighbor_function);
/// assert_eq!(cc, HashSet::from([2, 3, 5, 4, 6]));
///
/// let cc = find_cc(5, neighbor_function);
/// assert_eq!(cc, HashSet::from([2, 3, 5, 4, 6]));
///
/// let cc = find_cc(6, neighbor_function);
/// assert_eq!(cc, HashSet::from([6]));
///
/// let cc = find_cc(7, neighbor_function);
/// assert_eq!(cc, HashSet::from([7]));
///
/// let neighbor_function_undirected = |node: u32| -> Vec<u32> {
///     match node {
///         1 => vec![2, 3],
///         2 => vec![1, 3, 4],
///         3 => vec![1, 2, 5],
///         4 => vec![2, 5, 6],
///         5 => vec![3, 4],
///         6 => vec![5],
///         _ => vec![],
///     }
/// };
///
/// let cc = find_cc(1, neighbor_function_undirected);
/// assert_eq!(cc, HashSet::from([1, 2, 3, 4, 5, 6]));
///
/// let cc = find_cc(2, neighbor_function_undirected);
/// assert_eq!(cc, HashSet::from([1, 2, 3, 4, 5, 6]));
///
/// let cc = find_cc(3, neighbor_function_undirected);
/// assert_eq!(cc, HashSet::from([1, 2, 3, 4, 5, 6]));
///
/// let cc = find_cc(4, neighbor_function_undirected);
/// assert_eq!(cc, HashSet::from([1, 2, 3, 4, 5, 6]));
///
/// let cc = find_cc(5, neighbor_function_undirected);
/// assert_eq!(cc, HashSet::from([1, 2, 3, 4, 5, 6]));
///
/// let cc = find_cc(6, neighbor_function_undirected);
/// assert_eq!(cc, HashSet::from([1, 2, 3, 4, 5, 6]));
///
/// let cc = find_cc(7, neighbor_function_undirected);
/// assert_eq!(cc, HashSet::from([7]));
///
/// ```
pub fn find_cc<T>(node: T, neighbor_function: impl Fn(T) -> Vec<T>) -> HashSet<T>
where
    T: Eq + Hash + Copy,
{
    pathfinding::directed::bfs::bfs_reach(node, |&x| neighbor_function(x)).collect()
}

// Should do this for each connected component (degree of freedom!)
pub fn two_color<T>(nodes: &[T], neighbor_function: impl Fn(T) -> Vec<T>) -> Option<(HashSet<T>, HashSet<T>)>
where
    T: Eq + Hash + Clone + Copy + Debug,
{
    let mut pool = nodes.to_vec();
    let mut color1 = HashSet::new();
    let mut color2 = HashSet::new();

    while let Some(s) = pool.pop() {
        let mut queue = vec![s];

        while let Some(node) = queue.pop() {
            pool.retain(|x| x != &node);
            if color1.contains(&node) || color2.contains(&node) {
                continue;
            }

            let neighbors = neighbor_function(node);

            if neighbors.iter().any(|x| color1.contains(x)) {
                if neighbors.iter().any(|x| color2.contains(x)) {
                    return None;
                }
                color2.insert(node);
            } else if neighbors.iter().any(|x| color2.contains(x)) {
                if neighbors.iter().any(|x| color1.contains(x)) {
                    return None;
                }

                color1.insert(node);
            } else {
                // Degree of freedom.
                color2.insert(node);
            }

            queue.extend(neighbors);
        }
    }
    Some((color1, color2))
}

pub fn topological_sort<T>(nodes: &[T], neighbor_function: impl Fn(T) -> Vec<T>) -> Option<Vec<T>>
where
    T: Eq + Hash + Clone + Copy,
{
    pathfinding::directed::topological_sort::topological_sort(nodes, |&x| neighbor_function(x)).ok()
}
