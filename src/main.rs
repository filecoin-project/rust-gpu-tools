use rust_gpu_tools::*;
use std::fs::File;
use std::io::prelude::*;

fn main() {
    for dev in Brand::Amd.get_devices() {
        println!("AMD: {:?}", dev);
    }
    for dev in Brand::Nvidia.get_devices() {
        println!("NVIDIA: {:?}", dev);
    }

    let dev = Brand::Nvidia.get_devices()[0];
    println!("{:?}", dev);
    let another_prog = Program::from_opencl(
        dev,
        "__kernel void main(uint a) { printf(\"Hello Donya! %u\\n\", a); }",
    );
    call_kernel!(another_prog, "main", 12, 12345);
}
