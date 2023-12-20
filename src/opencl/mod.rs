//! The OpenCL specific implementation of a [`Buffer`], [`Device`], [`Program`] and [`Kernel`].

pub(crate) mod utils;

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::mem;
use std::ptr;

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::error_codes::ClError;
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::CL_MEM_READ_WRITE;
use opencl3::types::CL_BLOCKING;

use log::debug;

use crate::device::{DeviceUuid, PciId, Vendor};
use crate::error::{GPUError, GPUResult};
use crate::LocalBuffer;

/// The lowest level identifier of an OpenCL device, it changes whenever a device is initialized.
#[allow(non_camel_case_types)]
pub type cl_device_id = opencl3::types::cl_device_id;

/// A Buffer to be used for sending and receiving data to/from the GPU.
#[derive(Debug)]
pub struct Buffer<T> {
    buffer: opencl3::memory::Buffer<u8>,
    /// The number of T-sized elements.
    length: usize,
    _phantom: std::marker::PhantomData<T>,
}

/// OpenCL specific device.
#[derive(Debug, Clone)]
pub struct Device {
    vendor: Vendor,
    name: String,
    /// The total memory of the GPU in bytes.
    memory: u64,
    /// The number of parallel compute units.
    compute_units: u32,
    /// Major and minor version of the compute capabilitiy (only available on Nvidia GPUs).
    compute_capability: Option<(u32, u32)>,
    pci_id: PciId,
    uuid: Option<DeviceUuid>,
    device: opencl3::device::Device,
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

    /// Returns the name of the GPU, e.g. "GeForce RTX 3090".
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

    /// Returns the major and minor version of the compute capability (only available on Nvidia
    /// GPUs).
    pub fn compute_capability(&self) -> Option<(u32, u32)> {
        self.compute_capability
    }

    /// Returns the PCI-ID of the GPU, see the [`PciId`] type for more information.
    pub fn pci_id(&self) -> PciId {
        self.pci_id
    }

    /// Returns the PCI-ID of the GPU if available, see the [`DeviceUuid`] type for more
    /// information.
    pub fn uuid(&self) -> Option<DeviceUuid> {
        self.uuid
    }

    /// Low-level access to the device identifier.
    ///
    /// It changes when the device is initialized and should only be used to interact with other
    /// libraries that work on the lowest OpenCL level.
    pub fn cl_device_id(&self) -> cl_device_id {
        self.device.id()
    }
}

/// Abstraction that contains everything to run an OpenCL kernel on a GPU.
///
/// The majority of methods are the same as [`crate::cuda::Program`], so you can write code using this
/// API, which will then work with OpenCL as well as CUDA kernels.
#[allow(rustdoc::broken_intra_doc_links)]
pub struct Program {
    device_name: String,
    queue: CommandQueue,
    context: Context,
    kernels_by_name: HashMap<String, opencl3::kernel::Kernel>,
}

impl Program {
    /// Returns the name of the GPU, e.g. "GeForce RTX 3090".
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Creates a program for a specific device from OpenCL source code.
    pub fn from_opencl(device: &Device, src: &str) -> GPUResult<Program> {
        debug!("Creating OpenCL program from source.");
        let cached = utils::cache_path(device, src)?;
        if std::path::Path::exists(&cached) {
            let bin = std::fs::read(cached)?;
            Program::from_binary(device, bin)
        } else {
            let context = Context::from_device(&device.device)?;
            debug!(
                "Building kernel ({}) from source…",
                cached.to_string_lossy()
            );
            let mut program = opencl3::program::Program::create_from_source(&context, src)?;
            if let Err(build_error) = program.build(context.devices(), "") {
                let log = program.get_build_log(context.devices()[0])?;
                return Err(GPUError::Opencl3(build_error, Some(log)));
            }
            debug!(
                "Building kernel ({}) from source: done.",
                cached.to_string_lossy()
            );
            let queue = CommandQueue::create_default(&context, 0)?;
            let kernels = opencl3::kernel::create_program_kernels(&program)?;
            let kernels_by_name = kernels
                .into_iter()
                .map(|kernel| {
                    let name = kernel.function_name()?;
                    Ok((name, kernel))
                })
                .collect::<Result<_, ClError>>()?;
            let prog = Program {
                device_name: device.name(),
                queue,
                context,
                kernels_by_name,
            };
            let binaries = program
                .get_binaries()
                .map_err(GPUError::ProgramInfoNotAvailable)?;
            std::fs::write(cached, binaries[0].clone())?;
            Ok(prog)
        }
    }

    /// Creates a program for a specific device from a compiled OpenCL binary.
    pub fn from_binary(device: &Device, bin: Vec<u8>) -> GPUResult<Program> {
        debug!("Creating OpenCL program from binary.");
        let context = Context::from_device(&device.device)?;
        let bins = vec![&bin[..]];
        let mut program = unsafe {
            opencl3::program::Program::create_from_binary(&context, context.devices(), &bins)
        }?;
        if let Err(build_error) = program.build(context.devices(), "") {
            let log = program.get_build_log(context.devices()[0])?;
            return Err(GPUError::Opencl3(build_error, Some(log)));
        }
        let queue = CommandQueue::create_default(&context, 0)?;
        let kernels = opencl3::kernel::create_program_kernels(&program)?;
        let kernels_by_name = kernels
            .into_iter()
            .map(|kernel| {
                let name = kernel.function_name()?;
                Ok((name, kernel))
            })
            .collect::<Result<_, ClError>>()?;
        Ok(Program {
            device_name: device.name(),
            queue,
            context,
            kernels_by_name,
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
        let mut buff = opencl3::memory::Buffer::create(
            &self.context,
            CL_MEM_READ_WRITE,
            // The input length is the number of elements, but we create a `u8` buffer. Hence the
            // length needs to be the number of bytes.
            length * std::mem::size_of::<T>(),
            ptr::null_mut(),
        )?;

        // Write some data right-away. This makes a significant performance different.
        self.queue
            .enqueue_write_buffer(&mut buff, opencl3::types::CL_BLOCKING, 0, &[0u8], &[])?;

        Ok(Buffer::<T> {
            buffer: buff,
            length,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Creates a new buffer on the GPU and initializes with the given slice.
    pub fn create_buffer_from_slice<T>(&self, slice: &[T]) -> GPUResult<Buffer<T>> {
        // The underlying buffer is `u8`, hence we need the number of bytes.
        let bytes_len = mem::size_of_val(slice);

        let mut buffer = unsafe {
            opencl3::memory::Buffer::create(
                &self.context,
                CL_MEM_READ_WRITE,
                bytes_len,
                ptr::null_mut(),
            )?
        };
        // Transmuting types is safe as long a sizes match.
        let bytes = unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const T as *const u8, bytes_len)
        };
        // Write some data right-away. This makes a significant performance different.
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, &[0u8], &[])?;
            self.queue
                .enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, bytes, &[])?;
        };

        Ok(Buffer::<T> {
            buffer,
            length: slice.len(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Returns a kernel.
    ///
    /// The `global_work_size` does *not* follow the OpenCL definition. It is *not* the total
    /// number of threads. Instead it follows CUDA's definition and is the number of
    /// `local_work_size` sized thread groups. So the total number of threads is
    /// `global_work_size * local_work_size`.
    pub fn create_kernel(
        &self,
        name: &str,
        global_work_size: usize,
        local_work_size: usize,
    ) -> GPUResult<Kernel> {
        let kernel = self
            .kernels_by_name
            .get(name)
            .ok_or_else(|| GPUError::KernelNotFound(name.to_string()))?;
        let mut builder = ExecuteKernel::new(kernel);
        builder.set_global_work_size(global_work_size * local_work_size);
        builder.set_local_work_size(local_work_size);
        Ok(Kernel {
            builder,
            queue: &self.queue,
            num_local_buffers: 0,
        })
    }

    /// Puts data from an existing buffer onto the GPU.
    pub fn write_from_buffer<T>(
        &self,
        // From Rust's perspective, this buffer doesn't need to be mutable. But the sub-buffer is
        // mutating the buffer, so it really should be.
        buffer: &mut Buffer<T>,
        data: &[T],
    ) -> GPUResult<()> {
        assert!(data.len() <= buffer.length, "Buffer is too small");

        // It is safe as long as the sizes match.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const T as *const u8,
                mem::size_of_val(data),
            )
        };
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut buffer.buffer, CL_BLOCKING, 0, bytes, &[])?;
        }
        Ok(())
    }

    /// Reads data from the GPU into an existing buffer.
    pub fn read_into_buffer<T>(&self, buffer: &Buffer<T>, data: &mut [T]) -> GPUResult<()> {
        assert!(data.len() <= buffer.length, "Buffer is too small");

        // It is safe as long as the sizes match.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut T as *mut u8,
                mem::size_of_val(data),
            )
        };
        unsafe {
            self.queue
                .enqueue_read_buffer(&buffer.buffer, CL_BLOCKING, 0, bytes, &[])?;
        };
        Ok(())
    }

    /// Run some code in the context of the program.
    ///
    /// It takes the program as a parameter, so that we can use the same function body, for both
    /// the OpenCL and the CUDA code path. The only difference is the type of the program.
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
/// needed. This trait enables automatic coversions, so that any type implementing it can be
/// passed into a [`Kernel`].
pub trait KernelArgument {
    /// Apply the kernel argument to the kernel.
    fn push(&self, kernel: &mut Kernel);
}

impl<T> KernelArgument for Buffer<T> {
    fn push(&self, kernel: &mut Kernel) {
        unsafe {
            kernel.builder.set_arg(&self.buffer);
        }
    }
}

impl KernelArgument for i32 {
    fn push(&self, kernel: &mut Kernel) {
        unsafe {
            kernel.builder.set_arg(self);
        }
    }
}

impl KernelArgument for u32 {
    fn push(&self, kernel: &mut Kernel) {
        unsafe {
            kernel.builder.set_arg(self);
        }
    }
}

impl<T> KernelArgument for LocalBuffer<T> {
    fn push(&self, kernel: &mut Kernel) {
        unsafe {
            kernel
                .builder
                .set_arg_local_buffer(self.length * std::mem::size_of::<T>());
        }
        kernel.num_local_buffers += 1;
    }
}

/// A kernel that can be executed.
#[derive(Debug)]
pub struct Kernel<'a> {
    /// The underlying kernel builder.
    pub builder: ExecuteKernel<'a>,
    queue: &'a CommandQueue,
    /// There can only be a single [`LocalBuffer`] as parameter due to CUDA restrictions. This
    /// counts them, so that there can be an error if there are more `LocalBuffer` arguments.
    num_local_buffers: u8,
}

impl<'a> Kernel<'a> {
    /// Set a kernel argument.
    ///
    /// The arguments must live as long as the kernel. Hence make sure they are not dropped as
    /// long as the kernel is in use.
    ///
    /// Example where this behaviour is enforced and leads to a compile-time error:
    ///
    /// ```compile_fail
    /// use rust_gpu_tools::opencl::Program;
    ///
    /// fn would_break(program: &Program) {
    ///    let data = vec![1, 2, 3, 4];
    ///    let buffer = program.create_buffer_from_slice(&data).unwrap();
    ///    let kernel = program.create_kernel("my_kernel", 4, 256).unwrap();
    ///    let kernel = kernel.arg(&buffer);
    ///    // This drop wouldn't error if the arguments wouldn't be bound to the kernels lifetime.
    ///    drop(buffer);
    ///    kernel.run().unwrap();
    /// }
    /// ```
    pub fn arg<T: KernelArgument>(mut self, t: &'a T) -> Self {
        t.push(&mut self);
        self
    }

    /// Actually run the kernel.
    pub fn run(mut self) -> GPUResult<()> {
        if self.num_local_buffers > 1 {
            return Err(GPUError::Generic(
                "There cannot be more than one `LocalBuffer`.".to_string(),
            ));
        }
        unsafe {
            self.builder.enqueue_nd_range(self.queue)?;
        }
        Ok(())
    }
}
