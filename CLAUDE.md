# Metal Backend for rust-gpu-tools

This document contains build/lint/test commands and code style guidelines for the Metal backend implementation.

## Build Commands

```bash
# Build with only Metal support
cargo build --features=metal --no-default-features

# Build with Metal, CUDA, and OpenCL support
cargo build --features=metal,cuda,opencl

# Run tests with Metal support
cargo test --features=metal

# Run Metal example
cargo run --example metal/add --features=metal
```

## Thread Safety Considerations

The Metal backend has some thread safety constraints that differ from CUDA and OpenCL:

1. Metal devices aren't `Send`/`Sync`, so our current implementation uses `metal::Device::system_default()` 
   as a workaround for thread safety issues.

2. In a production implementation, this would need to be improved to properly handle device references 
   in a thread-safe manner.

## Implementation Notes

The Metal backend follows the same APIs as CUDA and OpenCL backends, including:

- `Device`: Represents a Metal GPU device
- `Program`: Runs Metal compute programs
- `Buffer`: Manages GPU memory
- `Kernel`: Configures and runs Metal kernels

The implementation is designed to be feature-compatible with the existing backends while respecting 
Metal's unique constraints and APIs.

## Known Limitations

- Only the system default Metal device is used (simplified implementation)
- PCI IDs for Metal devices are created from a hash of the device name
- No UUID support for Metal devices
- Limited thread safety

## Future Improvements

- Implement proper thread safety for Metal devices
- Add support for multiple Metal devices
- Improve device detection and properties accuracy
- Add more comprehensive tests and examples