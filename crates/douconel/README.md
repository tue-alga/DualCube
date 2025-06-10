# Douconel
Rust implementation for storing a triangular mesh as DCEL. Uses indices instead of pointers for **safety**. Features including but not limited to:
1. constructing DCEL
2. importing a (triangular) mesh from stl and obj and storing it as DCEL
3. exporting a DCEL to stl or obj
4. basic DCEL and geometry operationms
6. some compatibility with petgraph (converting to petgraph graph) and bevy (converting to renderable mesh)

Other types of graphs may also be constructed with the DCEL data structure, but are essentially untested with this implementation. Some operations may only yield correct (and safe) results for triangular meshes (faces of strictly degree 3). 
