# rust-gpu-tools [![Crates.io](https://img.shields.io/crates/v/rust-gpu-tools.svg)](https://crates.io/crates/rust-gpu-tools)

An abstraction library to run kernels on both CUDA and OpenCL.

## Example

You need to write the code that interacts with the GPU only once. Below is such code that runs a
kernel on CUDA and/or OpenCL. For a full working example, please see the [`examples`](examples)
directory. You can run it via `cargo run --example add`.

```rust
let closures = program_closures!(|program, _args| -> Result<Vec<u32>, GPUError> {
    // Make sure the input data has the same length.
    assert_eq!(aa.len(), bb.len());
    let length = aa.len();

    // Copy the data to the GPU.
    let aa_buffer = program.create_buffer_from_slice(&aa)?;
    let bb_buffer = program.create_buffer_from_slice(&bb)?;

    // The result buffer has the same length as the input buffers.
    let result_buffer = unsafe { program.create_buffer::<u32>(length)? };

    // Get the kernel.
    let kernel = program.create_kernel("add", 8, 4)?;

    // Execute the kernel.
    kernel
        .arg(&(length as u32))
        .arg(&aa_buffer)
        .arg(&bb_buffer)
        .arg(&result_buffer)
        .run()?;

    // Get the resulting data.
    let mut result = vec![0u32; length];
    program.read_into_buffer(&result_buffer, &mut result)?;

    Ok(result)
});
```


## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
