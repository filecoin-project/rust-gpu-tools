//! The Metal specific implementation of a [`Buffer`], [`Device`], [`Program`] and [`Kernel`].

pub(crate) mod utils;

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::mem;
use log::debug;
use metal::{self, CommandQueue, MTLResourceOptions};

use crate::device::{DeviceUuid, PciId, Vendor};
use crate::error::{GPUError, GPUResult};
use crate::LocalBuffer;

/// A Buffer to be used for sending and receiving data to/from the GPU.
#[derive(Debug)]
pub struct Buffer<T> {
    buffer: metal::Buffer,
    /// The number of T-sized elements.
    length: usize,
    _phantom: std::marker::PhantomData<T>,
}

/// Metal specific device.
#[derive(Debug, Clone)]
pub struct Device {
    vendor: Vendor,
    name: String,
    /// The total memory of the GPU in bytes.
    memory: u64,
    /// Number of compute units.
    compute_units: u32,
    pci_id: PciId,
    uuid: Option<DeviceUuid>,
    device_name: String,
}

impl Hash for Device {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vendor.hash(state);
        self.name.hash(state);
        self.memory.hash(state);
        self.pci_id.hash(state);
        self.uuid.hash(state);
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        self.vendor == other.vendor
            && self.name == other.name
            && self.memory == other.memory
            && self.pci_id == other.pci_id
            && self.uuid == other.uuid
    }
}

impl Eq for Device {}

impl Device {
    /// Returns the [`Vendor`] of the GPU.
    pub fn vendor(&self) -> Vendor {
        self.vendor
    }

    /// Returns the name of the GPU, e.g. "Apple M1 Pro".
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Returns the memory of the GPU in bytes.
    pub fn memory(&self) -> u64 {
        self.memory
    }

    /// Returns the number of compute units of the GPU.
    pub fn compute_units(&self) -> u32 {
        self.compute_units
    }

    /// Returns the PCI-ID of the GPU, see the [`PciId`] type for more information.
    pub fn pci_id(&self) -> PciId {
        self.pci_id
    }

    /// Returns the UUID of the GPU if available, see the [`DeviceUuid`] type for more
    /// information.
    pub fn uuid(&self) -> Option<DeviceUuid> {
        self.uuid
    }

    /// Returns a description of the Metal device.
    pub fn device_description(&self) -> &str {
        &self.device_name
    }
}

/// Abstraction that contains everything to run a Metal kernel on a GPU.
///
/// The majority of methods are the same as [`crate::opencl::Program`] and [`crate::cuda::Program`], so you can write code using this
/// API, which will then work with OpenCL, CUDA, and Metal kernels.
pub struct Program {
    device_name: String,
    queue: CommandQueue,
    functions: HashMap<String, metal::Function>,
    library: metal::Library,
}

impl Program {
    /// Returns the name of the GPU, e.g. "Apple M1 Pro".
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Creates a program for a specific device from Metal source code.
    pub fn from_metal(_device: &Device, src: &str) -> GPUResult<Program> {
        debug!("Creating Metal program from source.");
        
        // This is a simplified implementation due to thread-safety constraints
        // In a real implementation, we would use the actual Metal device
        let device = metal::Device::system_default().unwrap();
        
        let options = metal::CompileOptions::new();
        let library = device.new_library_with_source(src, &options)
            .map_err(|err| GPUError::MetalCompile(err.to_string()))?;
        
        let queue = device.new_command_queue();
        
        // Pre-load all functions from the library
        let function_names = library.function_names();
        let mut functions = HashMap::new();
        
        for name in function_names.iter() {
            if let Ok(function) = library.get_function(name, None) {
                functions.insert(name.to_string(), function);
            }
        }
        
        Ok(Program {
            device_name: device.name().to_string(),
            queue,
            functions,
            library,
        })
    }
    
    /// Creates a program for a specific device from a compiled Metal binary (metallib).
    pub fn from_binary(_device: &Device, bin: &[u8]) -> GPUResult<Program> {
        debug!("Creating Metal program from binary.");
        
        // This is a simplified implementation due to thread-safety constraints
        // In a real implementation, we would use the actual Metal device
        let device = metal::Device::system_default().unwrap();
        
        let library = device.new_library_with_data(bin)
            .map_err(|err| GPUError::MetalCompile(err.to_string()))?;
        
        let queue = device.new_command_queue();
        
        // Pre-load all functions from the library
        let function_names = library.function_names();
        let mut functions = HashMap::new();
        
        for name in function_names.iter() {
            if let Ok(function) = library.get_function(name, None) {
                functions.insert(name.to_string(), function);
            }
        }
        
        Ok(Program {
            device_name: device.name().to_string(),
            queue,
            functions,
            library,
        })
    }

    /// Creates a new buffer that can be used for input/output with the GPU.
    ///
    /// The `length` is the number of elements to create.
    ///
    /// It is usually used to create buffers that are initialized by the GPU. If you want to
    /// directly transfer data from the host to the GPU, you would use the safe
    /// [`Program::create_buffer_from_slice`] instead.
    ///
    /// # Safety
    ///
    /// This function isn't actually unsafe, it's marked as `unsafe` due to the CUDA version of it,
    /// where it is unsafe. This is done to have symmetry between both APIs.
    pub unsafe fn create_buffer<T>(&self, length: usize) -> GPUResult<Buffer<T>> {
        assert!(length > 0);
        
        // Get the Metal device from the system to ensure we have the right device
        let device = metal::Device::system_default().unwrap();
        let buffer = device.new_buffer(
            (length * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared
        );
        
        Ok(Buffer::<T> {
            buffer,
            length,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Creates a new buffer on the GPU and initializes with the given slice.
    pub fn create_buffer_from_slice<T>(&self, slice: &[T]) -> GPUResult<Buffer<T>> {
        // Calculate the size in bytes
        let bytes_len = mem::size_of_val(slice);
        
        // Get the Metal device from the system to ensure we have the right device
        let device = metal::Device::system_default().unwrap();
        
        // Create a Metal buffer with the correct size
        let buffer = device.new_buffer_with_data(
            slice.as_ptr() as *const _,
            bytes_len as u64,
            MTLResourceOptions::StorageModeShared
        );
        
        Ok(Buffer::<T> {
            buffer,
            length: slice.len(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Returns a kernel.
    ///
    /// The `global_work_size` follows CUDA's definition and is the number of
    /// `local_work_size` sized thread groups. So the total number of threads is
    /// `global_work_size * local_work_size`.
    pub fn create_kernel(
        &self,
        name: &str,
        global_work_size: usize,
        local_work_size: usize,
    ) -> GPUResult<Kernel> {
        let function = self.functions.get(name)
            .ok_or_else(|| GPUError::KernelNotFound(name.to_string()))?;
        
        // Get the device from the system
        let device = metal::Device::system_default().unwrap();
        let pipeline_state = device.new_compute_pipeline_state_with_function(function)
            .map_err(|err| GPUError::Metal(err.to_string()))?;
        
        Ok(Kernel {
            pipeline_state,
            global_work_size,
            local_work_size,
            queue: self.queue.clone(),
            args: Vec::new(),
            num_local_buffers: 0,
        })
    }

    /// Puts data from an existing buffer onto the GPU.
    pub fn write_from_buffer<T>(
        &self,
        buffer: &mut Buffer<T>,
        data: &[T],
    ) -> GPUResult<()> {
        assert!(data.len() <= buffer.length, "Buffer is too small");
        
        let bytes_len = mem::size_of_val(data);
        let dest_ptr = buffer.buffer.contents() as *mut u8;
        
        unsafe {
            let src_ptr = data.as_ptr() as *const u8;
            std::ptr::copy_nonoverlapping(src_ptr, dest_ptr, bytes_len);
        }
        
        Ok(())
    }

    /// Reads data from the GPU into an existing buffer.
    pub fn read_into_buffer<T>(&self, buffer: &Buffer<T>, data: &mut [T]) -> GPUResult<()> {
        assert!(data.len() <= buffer.length, "Buffer is too small");
        
        let bytes_len = mem::size_of_val(data);
        let src_ptr = buffer.buffer.contents() as *const u8;
        
        unsafe {
            let dest_ptr = data.as_mut_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src_ptr, dest_ptr, bytes_len);
        }
        
        Ok(())
    }

    /// Run some code in the context of the program.
    ///
    /// It takes the program as a parameter, so that we can use the same function body, for both
    /// the OpenCL, CUDA, and Metal code paths. The only difference is the type of the program.
    pub fn run<F, R, E, A>(&self, fun: F, arg: A) -> Result<R, E>
    where
        F: FnOnce(&Self, A) -> Result<R, E>,
        E: From<GPUError>,
    {
        fun(self, arg)
    }
}

/// Abstraction for kernel arguments.
///
/// The kernel doesn't support being called with custom types, hence some conversion might be
/// needed. This trait enables automatic conversions, so that any type implementing it can be
/// passed into a [`Kernel`].
pub trait KernelArgument {
    /// Apply the kernel argument to the kernel.
    fn push(&self, kernel: &mut Kernel, index: usize);
}

impl<T> KernelArgument for Buffer<T> {
    fn push(&self, kernel: &mut Kernel, index: usize) {
        kernel.args.push(KernelArg::Buffer(self.buffer.clone(), index));
    }
}

impl KernelArgument for i32 {
    fn push(&self, kernel: &mut Kernel, index: usize) {
        kernel.args.push(KernelArg::Int(*self, index));
    }
}

impl KernelArgument for u32 {
    fn push(&self, kernel: &mut Kernel, index: usize) {
        kernel.args.push(KernelArg::UInt(*self, index));
    }
}

impl<T> KernelArgument for LocalBuffer<T> {
    fn push(&self, kernel: &mut Kernel, index: usize) {
        kernel.num_local_buffers += 1;
        kernel.args.push(KernelArg::ThreadgroupMemory(
            self.length * std::mem::size_of::<T>(),
            index
        ));
    }
}

/// Represents different types of kernel arguments
#[derive(Debug)]
enum KernelArg {
    Buffer(metal::Buffer, usize),
    Int(i32, usize),
    UInt(u32, usize),
    ThreadgroupMemory(usize, usize),
}

/// A kernel that can be executed.
#[derive(Debug)]
pub struct Kernel {
    pipeline_state: metal::ComputePipelineState,
    global_work_size: usize,
    local_work_size: usize,
    queue: CommandQueue,
    args: Vec<KernelArg>,
    /// There can only be a single [`LocalBuffer`] as parameter due to CUDA restrictions. This
    /// counts them, so that there can be an error if there are more `LocalBuffer` arguments.
    num_local_buffers: u8,
}

impl Kernel {
    /// Set a kernel argument.
    ///
    /// The arguments must live as long as the kernel. Hence make sure they are not dropped as
    /// long as the kernel is in use.
    pub fn arg<T: KernelArgument>(mut self, t: &T) -> Self {
        let index = self.args.len();
        t.push(&mut self, index);
        self
    }

    /// Actually run the kernel.
    pub fn run(self) -> GPUResult<()> {
        if self.num_local_buffers > 1 {
            return Err(GPUError::Generic(
                "There cannot be more than one `LocalBuffer`.".to_string(),
            ));
        }
        
        let command_buffer = self.queue.new_command_buffer();
        let command_encoder = command_buffer.new_compute_command_encoder();
        
        command_encoder.set_compute_pipeline_state(&self.pipeline_state);
        
        // Set all arguments
        for arg in &self.args {
            match arg {
                KernelArg::Buffer(buffer, index) => {
                    command_encoder.set_buffer(*index as u64, Some(buffer), 0);
                }
                KernelArg::Int(value, index) => {
                    command_encoder.set_bytes(
                        *index as u64,
                        std::mem::size_of::<i32>() as u64,
                        value as *const _ as *const _
                    );
                }
                KernelArg::UInt(value, index) => {
                    command_encoder.set_bytes(
                        *index as u64,
                        std::mem::size_of::<u32>() as u64,
                        value as *const _ as *const _
                    );
                }
                KernelArg::ThreadgroupMemory(size, index) => {
                    command_encoder.set_threadgroup_memory_length(*index as u64, *size as u64);
                }
            }
        }
        
        // Calculate the grid and threadgroup sizes
        let grid_size = metal::MTLSize {
            width: self.global_work_size as u64 * self.local_work_size as u64,
            height: 1,
            depth: 1
        };
        
        let threadgroup_size = metal::MTLSize {
            width: self.local_work_size as u64,
            height: 1,
            depth: 1
        };
        
        // Dispatch the compute kernel
        command_encoder.dispatch_threads(grid_size, threadgroup_size);
        command_encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::utils;

    #[test]
    fn test_metal_device_list() {
        // Skip test if running CI on non-Apple hardware
        if cfg!(not(target_os = "macos")) {
            return;
        }
        
        let devices = utils::build_device_list();
        // Ensure we found at least one Metal device on macOS
        if cfg!(target_os = "macos") {
            assert!(!devices.is_empty(), "No Metal devices found on macOS");
            
            // Print found devices for debugging
            for device in &devices {
                println!("Found Metal device: {} ({} bytes, {} compute units)",
                         device.name(), device.memory(), device.compute_units());
            }
        }
    }
}