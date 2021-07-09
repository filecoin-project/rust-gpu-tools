pub(crate) mod utils;

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ptr;

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::error_codes::ClError;
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::CL_MEM_READ_WRITE;
use opencl3::program::ProgramInfo::CL_PROGRAM_BINARIES;
use opencl3::types::CL_BLOCKING;

use crate::device::{DeviceUuid, PciId, Vendor};
use crate::error::{GPUError, GPUResult};

#[allow(non_camel_case_types)]
pub type cl_device_id = opencl3::types::cl_device_id;

pub struct Buffer<T> {
    buffer: opencl3::memory::Buffer<T>,
    length: usize,
}

#[derive(Debug, Clone)]
pub struct Device {
    vendor: Vendor,
    name: String,
    memory: u64,
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

    pub fn cl_device_id(&self) -> cl_device_id {
        self.device.id()
    }
}

pub struct Program {
    device_name: String,
    queue: CommandQueue,
    context: Context,
    kernels_by_name: HashMap<String, opencl3::kernel::Kernel>,
}

impl Program {
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn from_opencl(device: &Device, src: &str) -> GPUResult<Program> {
        let cached = utils::cache_path(device, src)?;
        if std::path::Path::exists(&cached) {
            let bin = std::fs::read(cached)?;
            Program::from_binary(device, bin)
        } else {
            let context = Context::from_device(&device.device)?;
            let mut program = opencl3::program::Program::create_from_source(&context, src)?;
            if let Err(build_error) = program.build(context.devices(), "") {
                let log = program.get_build_log(context.devices()[0])?;
                return Err(GPUError::Opencl3(build_error, Some(log)));
            }
            let queue = CommandQueue::create(&context, context.default_device(), 0)?;
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
                .map_err(|_| GPUError::ProgramInfoNotAvailable(CL_PROGRAM_BINARIES))?;
            std::fs::write(cached, binaries[0].clone())?;
            Ok(prog)
        }
    }

    pub fn from_binary(device: &Device, bin: Vec<u8>) -> GPUResult<Program> {
        let context = Context::from_device(&device.device)?;
        let bins = vec![&bin[..]];
        let mut program =
            opencl3::program::Program::create_from_binary(&context, context.devices(), &bins)?;
        if let Err(build_error) = program.build(context.devices(), "") {
            let log = program.get_build_log(context.devices()[0])?;
            return Err(GPUError::Opencl3(build_error, Some(log)));
        }
        let queue = CommandQueue::create(&context, context.default_device(), 0)?;
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

    pub fn create_buffer<T>(&self, length: usize) -> GPUResult<Buffer<T>> {
        assert!(length > 0);
        let buff = opencl3::memory::Buffer::create(
            &self.context,
            CL_MEM_READ_WRITE,
            length,
            ptr::null_mut(),
        )?;

        Ok(Buffer::<T> {
            buffer: buff,
            length,
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
        let mut builder = ExecuteKernel::new(&kernel);
        builder.set_global_work_size(global_work_size * local_work_size);
        builder.set_local_work_size(local_work_size);
        Ok(Kernel {
            builder,
            queue: &self.queue,
        })
    }

    pub fn write_from_buffer<T>(
        &self,
        buffer: &Buffer<T>,
        offset: usize,
        data: &[T],
    ) -> GPUResult<()> {
        assert!(offset + data.len() <= buffer.length, "Buffer is too small");

        let buff = buffer
            .buffer
            .create_sub_buffer(CL_MEM_READ_WRITE, offset, data.len())?;

        self.queue
            .enqueue_write_buffer(&buff, CL_BLOCKING, 0, data, &[])?;

        Ok(())
    }

    pub fn read_into_buffer<T>(
        &self,
        buffer: &Buffer<T>,
        offset: usize,
        data: &mut [T],
    ) -> GPUResult<()> {
        assert!(offset + data.len() <= buffer.length, "Buffer is too small");
        let buff = buffer
            .buffer
            .create_sub_buffer(CL_MEM_READ_WRITE, offset, data.len())?;

        self.queue
            .enqueue_read_buffer(&buff, CL_BLOCKING, 0, data, &[])?;

        Ok(())
    }

    /// Run some code in the context of the program
    ///
    /// It takes the program as a parameter, so that we can use the same function body, for both
    /// the OpenCL and the CUDA code path. The only difference is the type of the program.
    pub fn run<F, R, E>(&self, fun: F) -> Result<R, E>
    where
        F: FnOnce(&Self) -> Result<R, E>,
        E: From<GPUError>,
    {
        fun(self)
    }
}

pub trait KernelArgument<'a> {
    fn push(&self, kernel: &mut Kernel<'a>);
}

impl<'a, T> KernelArgument<'a> for Buffer<T> {
    fn push(&self, kernel: &mut Kernel<'a>) {
        kernel.builder.set_arg(&self.buffer);
    }
}

impl KernelArgument<'_> for i32 {
    fn push(&self, kernel: &mut Kernel) {
        kernel.builder.set_arg(self);
    }
}

impl KernelArgument<'_> for u32 {
    fn push(&self, kernel: &mut Kernel) {
        kernel.builder.set_arg(self);
    }
}

pub struct LocalBuffer<T> {
    length: usize,
    _phantom: std::marker::PhantomData<T>,
}
impl<T> LocalBuffer<T> {
    pub fn new(length: usize) -> Self {
        LocalBuffer::<T> {
            length,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> KernelArgument<'_> for LocalBuffer<T> {
    fn push(&self, kernel: &mut Kernel) {
        kernel
            .builder
            .set_arg_local_buffer::<T>(self.length * std::mem::size_of::<T>());
    }
}

#[derive(Debug)]
pub struct Kernel<'a> {
    builder: ExecuteKernel<'a>,
    queue: &'a CommandQueue,
}

impl<'a> Kernel<'a> {
    pub fn arg<T: KernelArgument<'a>>(mut self, t: &T) -> Self {
        t.push(&mut self);
        self
    }
    pub fn run(mut self) -> GPUResult<()> {
        self.builder.enqueue_nd_range(&self.queue)?;
        Ok(())
    }
}
