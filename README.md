# Project Installation Guide

## Prerequisites
Ensure Rust and Cargo are installed on your system. Use `rustup` for installation:
- **Unix Systems**: 
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **Windows Systems**: 
  Download and run [rustup-init.exe](https://static.rust-lang.org/rustup/dist/i686-pc-windows-msvc/rustup-init.exe)

To update Rust and Cargo to the latest version:
```bash
rustup update
```

## Cloning the Repository
Clone the project from GitHub:
```bash
git clone https://www.github.com/tue-alga/DualCube
```
If you already have the project, update it to the latest version:
```bash
git pull
```

## Running the Project
Compile and run the project using Cargo:
```bash
cargo run
```
For a faster compilation time with a less optimized build, use:
```bash
cargo run --profile dev
```
