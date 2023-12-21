//! The CUDA specific implementation of a [`Buffer`], [`Device`], [`Program`] and [`Kernel`].
//!
//! The current operation mode is synchronuous, in order to have higher safety gurarantees. All
//! operations happen on a single stream, which is synchronized after each operation. This is a
//! similar behaviour to CUDA's default stream. The default stream isn't used for two reasons:
//!
//!   1. RustaCUDA doesn't expose a higher level function to launch a kernel on the default stream
//!   2. There was a bug, when the default stream was used implicitly via RustaCUDA's synchronuous
//!   copy methods. To prevent such kind of bugs, be explicit which stream is used.

pub(crate) mod utils;

use std::convert::TryFrom;
use std::ffi::{c_void, CStr, CString};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;

use log::debug;
use rustacuda::memory::{AsyncCopyDestination, DeviceBuffer};
use rustacuda::stream::{Stream, StreamFlags};

use crate::device::{DeviceUuid, PciId, Vendor};
use crate::error::{GPUError, GPUResult};
use crate::LocalBuffer;

/// A Buffer to be used for sending and receiving data to/from the GPU.
#[derive(Debug)]
pub struct Buffer<T> {
    buffer: DeviceBuffer<u8>,
    /// The number of T-sized elements.
    length: usize,
    _phantom: std::marker::PhantomData<T>,
}

/// CUDA specific device.
#[derive(Debug, Clone)]
pub struct Device {
    vendor: Vendor,
    name: String,
    /// The total memory of the GPU in bytes.
    memory: u64,
    /// Number of streaming multiprocessors.
    compute_units: u32,
    /// The compute capability of the device, major and minor version.
    compute_capability: (u32, u32),
    pci_id: PciId,
    uuid: Option<DeviceUuid>,
    context: rustacuda::context::UnownedContext,
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

    /// Returns the major and minor version of compute capability of the GPU.
    pub fn compute_capability(&self) -> (u32, u32) {
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
}

/// Abstraction that contains everything to run a CUDA kernel on a GPU.
///
/// The majority of methods are the same as [`crate::opencl::Program`], so you can write code using this
/// API, which will then work with OpenCL as well as CUDA kernels.
// When compiled without the `opencl` feature, then the intra-doc link above will be broken.
#[allow(rustdoc::broken_intra_doc_links)]
#[derive(Debug)]
pub struct Program {
    context: rustacuda::context::UnownedContext,
    module: rustacuda::module::Module,
    stream: Stream,
    device_name: String,
}

impl Program {
    /// Returns the name of the GPU, e.g. "GeForce RTX 3090".
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Creates a program for a specific device from a compiled CUDA binary file.
    pub fn from_binary(device: &Device, filename: &CStr) -> GPUResult<Program> {
        debug!("Creating CUDA program from binary file.");
        rustacuda::context::CurrentContext::set_current(&device.context)?;
        let module = rustacuda::module::Module::load_from_file(filename).map_err(|err| {
            Self::pop_context();
            err
        })?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|err| {
            Self::pop_context();
            err
        })?;
        let prog = Program {
            module,
            stream,
            device_name: device.name(),
            context: device.context.clone(),
        };
        Self::pop_context();
        Ok(prog)
    }

    /// Creates a program for a specific device from a compiled CUDA binary.
    pub fn from_bytes(device: &Device, bytes: &[u8]) -> GPUResult<Program> {
        debug!("Creating CUDA program from bytes.");
        rustacuda::context::CurrentContext::set_current(&device.context)?;
        let module = rustacuda::module::Module::load_from_bytes(bytes).map_err(|err| {
            Self::pop_context();
            err
        })?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|err| {
            Self::pop_context();
            err
        })?;
        let prog = Program {
            module,
            stream,
            device_name: device.name(),
            context: device.context.clone(),
        };
        Self::pop_context();
        Ok(prog)
    }

    /// Creates a new buffer that can be used for input/output with the GPU.
    ///
    /// The `length` is the number of elements to create.
    ///
    /// It is usually used to create buffers that are initialized by the GPU. If you want to
    /// directly transfer data from the host to the GPU, you would use the safe
    /// [`Program::create_buffer_from_slice`] instead.
    ///
    /// ### Safety
    ///
    /// The buffer needs to be initalized (by the host with [`Program::write_from_buffer`]) or by
    /// the GPU) before it can be read via [`Program::read_into_buffer`].
    pub unsafe fn create_buffer<T>(&self, length: usize) -> GPUResult<Buffer<T>> {
        assert!(length > 0);
        // This is the unsafe call, the rest of the function is safe code.
        let buffer = DeviceBuffer::<u8>::uninitialized(length * std::mem::size_of::<T>())?;

        Ok(Buffer::<T> {
            buffer,
            length,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Creates a new buffer on the GPU and initializes with the given slice.
    pub fn create_buffer_from_slice<T>(&self, slice: &[T]) -> GPUResult<Buffer<T>> {
        // The number of bytes is used for the allocations.
        let bytes_len = mem::size_of_val(slice);

        // Transmuting types is safe as long a sizes match.
        let bytes = unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const T as *const u8, bytes_len)
        };

        // It is only unsafe as long as the buffer isn't initialized, but that's what we do next.
        let mut buffer = unsafe { DeviceBuffer::<u8>::uninitialized(bytes_len)? };
        // It is safe as we synchronize the stream after the call.
        unsafe { buffer.async_copy_from(bytes, &self.stream)? };
        self.stream.synchronize()?;

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
    pub fn create_kernel(&self, name: &str, gws: usize, lws: usize) -> GPUResult<Kernel> {
        let function_name = CString::new(name).expect("Kernel name must not contain nul bytes");
        let function = self.module.get_function(&function_name)?;

        Ok(Kernel {
            function,
            global_work_size: gws,
            local_work_size: lws,
            stream: &self.stream,
            args: Vec::new(),
        })
    }

    /// Puts data from an existing buffer onto the GPU.
    pub fn write_from_buffer<T>(&self, buffer: &mut Buffer<T>, data: &[T]) -> GPUResult<()> {
        assert!(data.len() <= buffer.length, "Buffer is too small");

        // Transmuting types is safe as long a sizes match.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const T as *const u8,
                mem::size_of_val(data),
            )
        };

        // It is safe as we synchronize the stream after the call.
        unsafe { buffer.buffer.async_copy_from(bytes, &self.stream)? };
        self.stream.synchronize()?;

        Ok(())
    }

    /// Reads data from the GPU into an existing buffer.
    pub fn read_into_buffer<T>(&self, buffer: &Buffer<T>, data: &mut [T]) -> GPUResult<()> {
        assert!(data.len() <= buffer.length, "Buffer is too small");

        // Transmuting types is safe as long a sizes match.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut T as *mut u8,
                mem::size_of_val(data),
            )
        };

        // It is safe as we synchronize the stream after the call.
        unsafe { buffer.buffer.async_copy_to(bytes, &self.stream)? };
        self.stream.synchronize()?;

        Ok(())
    }

    /// Run some code in the context of the program.
    ///
    /// It sets the correct contexts.
    ///
    /// It takes the program as a parameter, so that we can use the same function body, for both
    /// the OpenCL and the CUDA code path. The only difference is the type of the program.
    pub fn run<F, R, E, A>(&self, fun: F, arg: A) -> Result<R, E>
    where
        F: FnOnce(&Self, A) -> Result<R, E>,
        E: From<GPUError>,
    {
        rustacuda::context::CurrentContext::set_current(&self.context).map_err(Into::into)?;
        let result = fun(self, arg);
        Self::pop_context();
        result
    }

    /// Pop the current context.
    ///
    /// It panics as it's an unrecoverable error.
    fn pop_context() {
        rustacuda::context::ContextStack::pop().expect("Cannot remove context.");
    }
}

// TODO vmx 2021-07-07: Check if RustaCUDA types used in `Program` can be made `Send`, so that
// this manual `Send` implementation is no longer needed.
unsafe impl Send for Program {}

/// Abstraction for kernel arguments.
///
/// Kernel arguments implement this trait, so that they can be converted it into the correct
/// pointers needed by the actual kernel call.
pub trait KernelArgument {
    /// Converts into a C void pointer.
    fn as_c_void(&self) -> *mut c_void;

    /// Returns the shared memory size. This is usally 0, except for [`LocalBuffer`]s. This
    /// informations is used to allocate the memory correctly.
    fn shared_mem(&self) -> u32 {
        0
    }
}

impl<T> KernelArgument for Buffer<T> {
    fn as_c_void(&self) -> *mut c_void {
        &self.buffer as *const _ as _
    }
}

impl KernelArgument for i32 {
    fn as_c_void(&self) -> *mut c_void {
        self as *const _ as _
    }
}

impl KernelArgument for u32 {
    fn as_c_void(&self) -> *mut c_void {
        self as *const _ as _
    }
}

impl<T> KernelArgument for LocalBuffer<T> {
    // This is a hack: on CUDA kernels, you cannot have `__shared__` (`__local` in OpenCL lingo)
    // kernel parameters. Hence, just pass on an arbirtary valid pointer. It won't be used, so it
    // doesn't matter where it actually points to. A null pointer cannot be used as CUDA would
    // return an "invalid argument" error.
    fn as_c_void(&self) -> *mut c_void {
        self as *const _ as _
    }

    fn shared_mem(&self) -> u32 {
        u32::try_from(self.length * std::mem::size_of::<T>())
            .expect("__shared__ memory allocation is too big.")
    }
}

/// A kernel that can be executed.
pub struct Kernel<'a> {
    function: rustacuda::function::Function<'a>,
    global_work_size: usize,
    local_work_size: usize,
    stream: &'a Stream,
    args: Vec<&'a dyn KernelArgument>,
}

impl fmt::Debug for Kernel<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let args = self
            .args
            .iter()
            .map(|arg| (arg.as_c_void(), arg.shared_mem()))
            .collect::<Vec<_>>();
        f.debug_struct("Kernel")
            .field("function", &self.function)
            .field("global_work_size", &self.global_work_size)
            .field("local_work_size", &self.local_work_size)
            .field("stream", &self.stream)
            .field("args", &args)
            .finish()
    }
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
    /// use rust_gpu_tools::cuda::Program;
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
        self.args.push(t);
        self
    }

    /// Actually run the kernel.
    ///
    /// ### Panics
    ///
    /// Panics if the wrong number of arguments was provided.
    pub fn run(self) -> GPUResult<()> {
        // There can only be a single [`LocalBuffer`], due to CUDA restrictions.
        let shared_mem = self
            .args
            .iter()
            .try_fold(0, |acc, &arg| -> GPUResult<u32> {
                let mem = arg.shared_mem();
                match (mem, acc) {
                    // No new shared memory needs to be allocated.
                    (0, _) => Ok(acc),
                    // Some shared memory needs to be allocated.
                    (_, 0) => Ok(mem),
                    // There should be memory allocated more than once
                    (_, _) => Err(GPUError::Generic(
                        "There cannot be more than one `LocalBuffer`.".to_string(),
                    )),
                }
            })?;
        let args = self
            .args
            .iter()
            .map(|arg| arg.as_c_void())
            .collect::<Vec<_>>();
        // It is safe to launch the kernel as the arguments need to live when the kernel is called,
        // and the buffers are copied synchronuously. At the end of the execution, the underlying
        // stream is synchronized.
        unsafe {
            self.stream.launch(
                &self.function,
                self.global_work_size as u32,
                self.local_work_size as u32,
                shared_mem,
                &args,
            )?;
        };
        // Synchronize after the kernel execution, so that the underlying pointers can be
        // invalidated/dropped.
        self.stream.synchronize()?;
        Ok(())
    }
}
