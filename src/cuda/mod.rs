pub(crate) mod utils;

use std::ffi::{c_void, CStr, CString};
use std::hash::{Hash, Hasher};

use rustacuda::memory::AsyncCopyDestination;

use crate::device::{DeviceUuid, PciId, Vendor};
use crate::error::{GPUError, GPUResult};

pub struct Buffer<T> {
    // We cannot use `T` directly for the `DeviceBuffer` as `AsyncCopyDestination` is only
    // implemented for `u8`.
    buffer: rustacuda::memory::DeviceBuffer<u8>,
    length: usize,
    _phantom: std::marker::PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct Device {
    vendor: Vendor,
    name: String,
    memory: u64,
    pci_id: PciId,
    uuid: Option<DeviceUuid>,
    device: rustacuda::device::Device,
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
    pub fn vendor(&self) -> Vendor {
        self.vendor
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn memory(&self) -> u64 {
        self.memory
    }

    pub fn pci_id(&self) -> PciId {
        self.pci_id
    }

    pub fn uuid(&self) -> Option<DeviceUuid> {
        self.uuid
    }
}

pub struct Program {
    context: rustacuda::context::UnownedContext,
    module: rustacuda::module::Module,
    stream: rustacuda::stream::Stream,
    device_name: String,
}

impl Program {
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn from_binary(device: &Device, filename: &CStr) -> GPUResult<Program> {
        rustacuda::context::CurrentContext::set_current(&device.context)?;
        let module = rustacuda::module::Module::load_from_file(filename)?;
        let stream =
            rustacuda::stream::Stream::new(rustacuda::stream::StreamFlags::NON_BLOCKING, None)?;
        let prog = Program {
            module,
            stream,
            device_name: device.name(),
            context: device.context.clone(),
        };
        rustacuda::context::ContextStack::pop().expect("Cannot remove newly created context.");
        Ok(prog)
    }

    pub fn create_buffer<T>(&self, length: usize) -> GPUResult<Buffer<T>> {
        assert!(length > 0);
        let buffer = unsafe {
            rustacuda::memory::DeviceBuffer::<u8>::uninitialized(length * std::mem::size_of::<T>())?
        };

        Ok(Buffer::<T> {
            buffer,
            length,
            _phantom: std::marker::PhantomData,
        })
    }

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

    pub fn write_from_buffer<T>(
        &self,
        buffer: &mut Buffer<T>,
        offset: usize,
        data: &[T],
    ) -> GPUResult<()> {
        assert!(offset + data.len() <= buffer.length);
        unsafe {
            let bytes = std::slice::from_raw_parts(
                data.as_ptr() as *const T as *const u8,
                data.len() * std::mem::size_of::<T>(),
            );
            buffer.buffer.async_copy_from(bytes, &self.stream)?;
        };
        Ok(())
    }

    pub fn read_into_buffer<T>(
        &self,
        buffer: &Buffer<T>,
        offset: usize,
        data: &mut [T],
    ) -> GPUResult<()> {
        assert!(offset + data.len() <= buffer.length);

        unsafe {
            let bytes = std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut T as *mut u8,
                data.len() * std::mem::size_of::<T>(),
            );
            buffer.buffer.async_copy_to(bytes, &self.stream)?;
        };

        Ok(())
    }

    /// Run some code in the context of the program
    ///
    /// It sets the correct contexts and synchronizes the stream before returning.
    ///
    /// It takes the program as a parameter, so that we can use the same function body, for both
    /// the OpenCL and the CUDA code path. The only difference is the type of the program.
    pub fn run<F, R, E>(&self, fun: F) -> Result<R, E>
    where
        F: FnOnce(&Self) -> Result<R, E>,
        E: From<GPUError>,
    {
        rustacuda::context::CurrentContext::set_current(&self.context).map_err(Into::into)?;
        let result = fun(self);
        self.stream.synchronize().map_err(Into::into)?;
        rustacuda::context::ContextStack::pop().map_err(Into::into)?;
        result
    }
}

// TODO vmx 2021-07-07: Check if RustaCUDA types use in `Program` can be made `Send`, so that
// this manual `Send` implementation is no longer needed.
unsafe impl Send for Program {}

/// Kernel arguments implement this trait, so that we can convert it into the correct pointers
/// needed by the actual kernel call.
pub trait KernelArgument<'a> {
    fn as_c_void(&self) -> *mut c_void;
}

impl<'a, T> KernelArgument<'a> for Buffer<T> {
    fn as_c_void(&self) -> *mut c_void {
        &self.buffer as *const _ as _
    }
}

impl KernelArgument<'_> for i32 {
    fn as_c_void(&self) -> *mut c_void {
        self as *const _ as _
    }
}
impl KernelArgument<'_> for u32 {
    fn as_c_void(&self) -> *mut c_void {
        self as *const _ as _
    }
}

#[derive(Debug)]
pub struct Kernel<'a> {
    function: rustacuda::function::Function<'a>,
    global_work_size: usize,
    local_work_size: usize,
    stream: &'a rustacuda::stream::Stream,
    args: Vec<*mut c_void>,
}

impl<'a> Kernel<'a> {
    pub fn arg<T: KernelArgument<'a>>(mut self, t: &T) -> Self {
        self.args.push(t.as_c_void());
        self
    }

    pub fn run(self) -> GPUResult<()> {
        unsafe {
            self.stream.launch(
                &self.function,
                self.global_work_size as u32,
                self.local_work_size as u32,
                0,
                &self.args,
            )?;
        };
        Ok(())
    }
}
