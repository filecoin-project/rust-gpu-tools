mod error;
mod utils;

use std::convert::TryFrom;
use std::fmt;
use std::hash::{Hash, Hasher};

pub use error::{GPUError, GPUResult};

pub type PciId = u32;

#[allow(non_camel_case_types)]
pub type cl_device_id = ocl::ffi::cl_device_id;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Brand {
    Amd,
    Apple,
    Nvidia,
}

impl Brand {
    /// Returns a brand by name if it exists
    fn by_name(name: &str) -> Option<Self> {
        match name {
            "NVIDIA CUDA" => Some(Self::Nvidia),
            "AMD Accelerated Parallel Processing" => Some(Self::Amd),
            "Apple" => Some(Self::Apple),
            _ => None,
        }
    }
}

impl fmt::Display for Brand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let brand = match self {
            Brand::Nvidia => "NVIDIA CUDA",
            Brand::Amd => "AMD Accelerated Parallel Processing",
            Brand::Apple => "Apple",
        };
        write!(f, "{}", brand)
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

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct DeviceUuid([u8; utils::CL_UUID_SIZE_KHR]);

impl TryFrom<&str> for DeviceUuid {
    type Error = GPUError;

    fn try_from(value: &str) -> GPUResult<DeviceUuid> {
        let res = value
            .split('-')
            .map(|s| hex::decode(s).map_err(|_| GPUError::Uuid(value.to_string())))
            .collect::<GPUResult<Vec<_>>>()?;

        let res = res.into_iter().flatten().collect::<Vec<u8>>();

        if res.len() != utils::CL_UUID_SIZE_KHR {
            Err(GPUError::Uuid(value.to_string()))
        } else {
            let mut raw = [0u8; utils::CL_UUID_SIZE_KHR];
            raw.copy_from_slice(res.as_slice());
            Ok(DeviceUuid(raw))
        }
    }
}

impl TryFrom<String> for DeviceUuid {
    type Error = GPUError;

    fn try_from(value: String) -> GPUResult<DeviceUuid> {
        DeviceUuid::try_from(value.as_ref())
    }
}

impl fmt::Display for DeviceUuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use hex::encode;

        // formats the uuid the same way as clinfo does, as an example:
        // the output should looks like 46abccd6-022e-b783-572d-833f7104d05f
        write!(
            f,
            "{}-{}-{}-{}-{}",
            encode(&self.0[..4]),
            encode(&self.0[4..6]),
            encode(&self.0[6..8]),
            encode(&self.0[8..10]),
            encode(&self.0[10..])
        )
    }
}

impl fmt::Debug for DeviceUuid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct Device {
    brand: Brand,
    name: String,
    memory: u64,
    platform: ocl::Platform,
    pci_id: Option<PciId>,
    uuid: Option<DeviceUuid>,
    pub device: ocl::Device,
}

impl Hash for Device {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // hash both properties because a device might have set only one
        self.uuid.hash(state);
        self.pci_id.hash(state);
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        // A device might have set only one of the properties, hence compare both
        self.uuid == other.uuid && self.pci_id == other.pci_id
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
        utils::is_little_endian(self.device)
    }
    pub fn pci_id(&self) -> Option<PciId> {
        self.pci_id
    }
    pub fn uuid(&self) -> Option<DeviceUuid> {
        self.uuid
    }

    /// Return all available GPU devices of supported brands.
    pub fn all() -> Vec<&'static Device> {
        Self::all_iter().collect()
    }

    pub fn by_pci_id(pci_id: PciId) -> GPUResult<&'static Device> {
        Device::all_iter()
            .find(|d| match d.pci_id {
                Some(id) => pci_id == id,
                None => false,
            })
            .ok_or(GPUError::DeviceNotFound)
    }

    pub fn by_uuid(uuid: &DeviceUuid) -> GPUResult<&'static Device> {
        Device::all_iter()
            .find(|d| match d.uuid {
                Some(ref id) => id == uuid,
                None => false,
            })
            .ok_or(GPUError::DeviceNotFound)
    }

    pub fn cl_device_id(&self) -> ocl::ffi::cl_device_id {
        self.device.as_core().as_raw()
    }

    fn all_iter() -> impl Iterator<Item = &'static Device> {
        utils::DEVICES.iter()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GPUSelector {
    Uuid(DeviceUuid),
    PciId(u32),
    Index(usize),
}

impl GPUSelector {
    pub fn get_uuid(&self) -> Option<DeviceUuid> {
        self.get_device().and_then(|dev| dev.uuid)
    }

    pub fn get_pci_id(&self) -> Option<u32> {
        self.get_device().and_then(|dev| dev.pci_id)
    }

    pub fn get_device(&self) -> Option<&'static Device> {
        match self {
            GPUSelector::Uuid(uuid) => Device::all_iter().find(|d| d.uuid == Some(*uuid)),
            GPUSelector::PciId(pci_id) => Device::all_iter().find(|d| d.pci_id == Some(*pci_id)),
            GPUSelector::Index(index) => get_device_by_index(*index),
        }
    }

    pub fn get_key(&self) -> String {
        match self {
            GPUSelector::Uuid(uuid) => format!("Uuid: {}", uuid),
            GPUSelector::PciId(id) => format!("PciId: {}", id),
            GPUSelector::Index(idx) => {
                format!("Index: {}", idx)
            }
        }
    }
}

fn get_device_by_index(index: usize) -> Option<&'static Device> {
    Device::all_iter().nth(index)
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
            step /= 2;
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
        kernel.builder.arg(*self);
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
    use super::{Device, DeviceUuid};
    use std::convert::TryFrom;

    #[test]
    fn test_device_all() {
        let devices = Device::all();
        dbg!(&devices.len());
        println!("{:?}", devices);
    }

    #[test]
    fn test_uuid() {
        let test_uuid = "46abccd6-022e-b783-572d-833f7104d05f";
        let uuid = DeviceUuid::try_from(test_uuid).unwrap();
        assert_eq!(test_uuid, &uuid.to_string());

        // test wrong length uuid
        let bad_uuid = "46abccd6-022e-b783-572-833f7104d05f";
        let uuid = DeviceUuid::try_from(bad_uuid);
        assert!(uuid.is_err());

        // test invalid hex character
        let bad_uuid = "46abccd6-022e-b783-572d-833f7104d05h";
        let uuid = DeviceUuid::try_from(bad_uuid);
        assert!(uuid.is_err());
    }
}
