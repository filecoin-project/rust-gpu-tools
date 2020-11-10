mod error;
mod utils;

pub use error::*;
use sha2::{Digest, Sha256};
use std::fmt::Write;
use std::hash::{Hash, Hasher};

pub type BusId = u32;

#[allow(non_camel_case_types)]
pub type cl_device_id = ocl::ffi::cl_device_id;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Brand {
    Amd,
    Apple,
    Nvidia,
}

impl Brand {
    fn all() -> Vec<Brand> {
        vec![Brand::Nvidia, Brand::Amd, Brand::Apple]
    }
}

pub struct Buffer<T> {
    buffer: ocl::Buffer<u8>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Buffer<T> {
    pub fn length(&self) -> usize {
        self.buffer.len() / std::mem::size_of::<T>()
    }

    pub fn write_from(&mut self, offset: usize, data: &[T]) -> GPUResult<()> {
        assert!(offset + data.len() <= self.length());
        self.buffer
            .create_sub_buffer(
                None,
                offset * std::mem::size_of::<T>(),
                data.len() * std::mem::size_of::<T>(),
            )?
            .write(unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const T as *const u8,
                    data.len() * std::mem::size_of::<T>(),
                )
            })
            .enq()?;
        Ok(())
    }

    pub fn read_into(&self, offset: usize, data: &mut [T]) -> GPUResult<()> {
        assert!(offset + data.len() <= self.length());
        self.buffer
            .create_sub_buffer(
                None,
                offset * std::mem::size_of::<T>(),
                data.len() * std::mem::size_of::<T>(),
            )?
            .read(unsafe {
                std::slice::from_raw_parts_mut(
                    data.as_mut_ptr() as *mut T as *mut u8,
                    data.len() * std::mem::size_of::<T>(),
                )
            })
            .enq()?;
        Ok(())
    }
}

impl Brand {
    pub fn platform_name(&self) -> &'static str {
        match self {
            Brand::Nvidia => "NVIDIA CUDA",
            Brand::Amd => "AMD Accelerated Parallel Processing",
            Brand::Apple => "Apple",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Device {
    brand: Brand,
    name: String,
    memory: u64,
    bus_id: Option<BusId>,
    platform: ocl::Platform,
    pub device: ocl::Device,
}

impl Hash for Device {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bus_id.hash(state);
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        self.bus_id == other.bus_id
    }
}

impl Eq for Device {}

impl Device {
    pub fn brand(&self) -> Brand {
        self.brand
    }
    pub fn name(&self) -> String {
        self.name.clone()
    }
    pub fn memory(&self) -> u64 {
        self.memory
    }
    pub fn is_little_endian(&self) -> GPUResult<bool> {
        Ok(utils::is_little_endian(self.device)?)
    }
    pub fn bus_id(&self) -> Option<BusId> {
        self.bus_id
    }

    // Return all available GPU devices of supported brands, ordered by brand as
    // defined by Brand::all().
    pub fn all() -> GPUResult<Vec<Device>> {
        let mut all = Vec::new();
        for b in &Brand::all() {
            all.append(&mut Device::by_brand(*b)?);
        }
        Ok(all)
    }

    pub fn by_bus_id(bus_id: BusId) -> GPUResult<Device> {
        Device::all().and_then(|devs| {
            devs.into_iter()
                .find(|d| match d.bus_id {
                    Some(id) => bus_id == id,
                    None => false,
                })
                .ok_or(GPUError::DeviceNotFound)
        })
    }

    pub fn by_brand(brand: Brand) -> GPUResult<Vec<Device>> {
        match utils::find_platform(brand.platform_name())? {
            Some(plat) => {
                // List all GPU devices on `plat`.
                ocl::Device::list(plat, Some(ocl::core::DeviceType::GPU))?
                    .into_iter()
                    .filter(|d| {
                        // Only return available devices.
                        d.is_available().unwrap_or(false)
                    })
                    .map(|d| {
                        (|| -> GPUResult<Device> {
                            Ok(Device {
                                brand,
                                name: d.name()?,
                                memory: get_memory(d)?,
                                bus_id: utils::get_bus_id(d).ok(),
                                platform: plat,
                                device: d,
                            })
                        })()
                    })
                    .collect()
            }
            None => Ok(Vec::new()),
        }
    }
    pub fn cl_device_id(&self) -> ocl::ffi::cl_device_id {
        self.device.as_core().as_raw()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GPUSelector {
    BusId(u32),
    Index(usize),
}

impl GPUSelector {
    pub fn get_bus_id(&self) -> Option<u32> {
        match self {
            GPUSelector::BusId(bus_id) => Some(*bus_id),
            GPUSelector::Index(index) => get_device_bus_id_by_index(*index),
        }
    }

    pub fn get_device(&self) -> Option<Device> {
        match self {
            GPUSelector::BusId(_bus_id) => todo!(),
            GPUSelector::Index(index) => get_device_by_index(*index),
        }
    }

    pub fn get_key(&self) -> String {
        match self {
            GPUSelector::BusId(id) => format!("BusID: {}", id),
            GPUSelector::Index(idx) => {
                if let Some(id) = self.get_bus_id() {
                    format!("BusID: {}", id)
                } else {
                    format!("Index: {}", idx)
                }
            }
        }
    }
}

fn get_device_bus_id_by_index(index: usize) -> Option<BusId> {
    if let Some(device) = get_device_by_index(index) {
        device.bus_id
    } else {
        None
    }
}

fn get_device_by_index(index: usize) -> Option<Device> {
    match Device::all() {
        Ok(all_devices) => {
            if index < all_devices.len() {
                Some(all_devices[index].clone())
            } else {
                None
            }
        }
        Err(_) => None,
    }
}

pub fn get_memory(d: ocl::Device) -> GPUResult<u64> {
    match d.info(ocl::enums::DeviceInfo::GlobalMemSize)? {
        ocl::enums::DeviceInfoResult::GlobalMemSize(sz) => Ok(sz),
        _ => Err(GPUError::DeviceInfoNotAvailable(
            ocl::enums::DeviceInfo::GlobalMemSize,
        )),
    }
}

pub struct Program {
    device: Device,
    program: ocl::Program,
    queue: ocl::Queue,
}

impl Program {
    pub fn device(&self) -> Device {
        self.device.clone()
    }
    pub fn from_opencl(device: Device, src: &str) -> GPUResult<Program> {
        let cached = utils::cache_path(&device, src)?;
        if std::path::Path::exists(&cached) {
            let bin = std::fs::read(cached)?;
            Program::from_binary(device, bin)
        } else {
            let context = ocl::Context::builder()
                .platform(device.platform)
                .devices(device.device)
                .build()?;
            let program = ocl::Program::builder()
                .src(src)
                .devices(ocl::builders::DeviceSpecifier::Single(device.device))
                .build(&context)?;
            let queue = ocl::Queue::new(&context, device.device, None)?;
            let prog = Program {
                program,
                queue,
                device,
            };
            std::fs::write(cached, prog.to_binary()?)?;
            Ok(prog)
        }
    }
    pub fn from_binary(device: Device, bin: Vec<u8>) -> GPUResult<Program> {
        let context = ocl::Context::builder()
            .platform(device.platform)
            .devices(device.device)
            .build()?;
        let bins = vec![&bin[..]];
        let program = ocl::Program::builder()
            .binaries(&bins)
            .devices(ocl::builders::DeviceSpecifier::Single(device.device))
            .build(&context)?;
        let queue = ocl::Queue::new(&context, device.device, None)?;
        Ok(Program {
            device,
            program,
            queue,
        })
    }
    pub fn to_binary(&self) -> GPUResult<Vec<u8>> {
        match self.program.info(ocl::enums::ProgramInfo::Binaries)? {
            ocl::enums::ProgramInfoResult::Binaries(bins) => Ok(bins[0].clone()),
            _ => Err(GPUError::ProgramInfoNotAvailable(
                ocl::enums::ProgramInfo::Binaries,
            )),
        }
    }
    pub fn create_buffer<T>(&self, length: usize) -> GPUResult<Buffer<T>> {
        assert!(length > 0);
        let buff = ocl::Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .flags(ocl::MemFlags::new().read_write())
            .len(length * std::mem::size_of::<T>())
            .build()?;
        buff.write(&vec![0u8]).enq()?;
        Ok(Buffer::<T> {
            buffer: buff,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn create_buffer_flexible<T>(&self, max_length: usize) -> GPUResult<Buffer<T>> {
        let mut curr = max_length;
        let mut step = max_length / 2;
        let mut n = 1;
        while step > 0 && n < max_length {
            if self.create_buffer::<T>(curr).is_ok() {
                n = curr;
                curr = std::cmp::min(curr + step, max_length);
            } else {
                curr -= step;
            }
            step = step / 2;
        }
        self.create_buffer::<T>(n)
    }
    pub fn create_kernel(&self, name: &str, gws: usize, lws: Option<usize>) -> Kernel<'_> {
        let mut builder = ocl::Kernel::builder();
        builder.name(name);
        builder.program(&self.program);
        builder.queue(self.queue.clone());
        builder.global_work_size([gws]);
        if let Some(lws) = lws {
            builder.local_work_size([lws]);
        }
        Kernel::<'_> { builder }
    }
}

pub use ocl::OclPrm as Parameter;

pub trait KernelArgument<'a> {
    fn push(&self, kernel: &mut Kernel<'a>);
}

impl<'a, T> KernelArgument<'a> for &'a Buffer<T> {
    fn push(&self, kernel: &mut Kernel<'a>) {
        kernel.builder.arg(&self.buffer);
    }
}

impl<T: ocl::OclPrm> KernelArgument<'_> for T {
    fn push(&self, kernel: &mut Kernel) {
        kernel.builder.arg(self.clone());
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
            .arg_local::<u8>(self.length * std::mem::size_of::<T>());
    }
}

#[derive(Debug)]
pub struct Kernel<'a> {
    builder: ocl::builders::KernelBuilder<'a>,
}

impl<'a> Kernel<'a> {
    pub fn arg<T: KernelArgument<'a>>(mut self, t: T) -> Self {
        t.push(&mut self);
        self
    }
    pub fn run(self) -> GPUResult<()> {
        let kern = self.builder.build()?;
        unsafe {
            kern.enq()?;
        }
        Ok(())
    }
}

#[macro_export]
macro_rules! call_kernel {
    ($kernel:expr, $($arg:expr),*) => {{
        $kernel
        $(.arg($arg))*
        .run()
    }};
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_device_all() {
        for _ in 0..10 {
            let devices = Device::all().unwrap();
            dbg!(&devices.len());
        }
    }
}
