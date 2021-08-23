//! The OpenCL specific implementation of a [`Buffer`], [`Device`], [`Program`] and [`Kernel`].

mod error;
mod utils;

use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;
use std::ptr;

pub use error::{GPUError, GPUResult};

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{DeviceInfo::CL_DEVICE_ENDIAN_LITTLE, CL_UUID_SIZE_KHR};
use opencl3::error_codes::ClError;
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::CL_MEM_READ_WRITE;
use opencl3::program::ProgramInfo::CL_PROGRAM_BINARIES;
use opencl3::types::CL_BLOCKING;

const AMD_DEVICE_VENDOR_STRING: &str = "Advanced Micro Devices, Inc.";
const AMD_DEVICE_VENDOR_ID: u32 = 0x1002;
// For some reason integrated AMD cards on Apple don't have the usual vendor name and ID
const AMD_DEVICE_ON_APPLE_VENDOR_STRING: &str = "AMD";
const AMD_DEVICE_ON_APPLE_VENDOR_ID: u32 = 0x1021d00;
const NVIDIA_DEVICE_VENDOR_STRING: &str = "NVIDIA Corporation";
const NVIDIA_DEVICE_VENDOR_ID: u32 = 0x10de;

#[allow(non_camel_case_types)]
pub type cl_device_id = opencl3::types::cl_device_id;

// The PCI-ID is the combination of the PCI Bus ID and PCI Device ID.
///
/// It is the first two identifiers of e.g. `lspci`:
///
/// ```text
///     4e:00.0 VGA compatible controller
///     || └└-- Device ID
///     └└-- Bus ID
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Hash)]
pub struct PciId(u16);

impl From<u16> for PciId {
    fn from(id: u16) -> Self {
        Self(id)
    }
}

impl From<PciId> for u16 {
    fn from(id: PciId) -> Self {
        id.0
    }
}

/// Converts a PCI-ID formatted as Bus-ID:Device-ID, e.g. `e3:00`.
impl TryFrom<&str> for PciId {
    type Error = GPUError;

    fn try_from(pci_id: &str) -> GPUResult<Self> {
        let mut bytes = [0; mem::size_of::<u16>()];
        hex::decode_to_slice(pci_id.replace(":", ""), &mut bytes).map_err(|_| {
            GPUError::InvalidId(format!(
                "Cannot parse PCI ID, expected hex-encoded string formated as aa:bb, got {0}.",
                pci_id
            ))
        })?;
        let parsed = u16::from_be_bytes(bytes);
        Ok(Self(parsed))
    }
}

/// Formats the PCI-ID like `lspci`, Bus-ID:Device-ID, e.g. `e3:00`.
impl fmt::Display for PciId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = u16::to_be_bytes(self.0);
        write!(f, "{:02x}:{:02x}", bytes[0], bytes[1])
    }
}

/// A unique identifier based on UUID of the device.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
pub struct DeviceUuid([u8; CL_UUID_SIZE_KHR]);

impl From<[u8; CL_UUID_SIZE_KHR]> for DeviceUuid {
    fn from(uuid: [u8; CL_UUID_SIZE_KHR]) -> Self {
        Self(uuid)
    }
}

impl From<DeviceUuid> for [u8; CL_UUID_SIZE_KHR] {
    fn from(uuid: DeviceUuid) -> Self {
        uuid.0
    }
}

/// Converts a UUID formatted as aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,
/// e.g. 46abccd6-022e-b783-572d-833f7104d05f
impl TryFrom<&str> for DeviceUuid {
    type Error = GPUError;

    fn try_from(uuid: &str) -> GPUResult<Self> {
        let mut bytes = [0; CL_UUID_SIZE_KHR];
        hex::decode_to_slice(uuid.replace("-", ""), &mut bytes)
            .map_err(|_| {
                GPUError::InvalidId(format!("Cannot parse UUID, expected hex-encoded string formated as aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, got {0}.", uuid))
            })?;
        Ok(Self(bytes))
    }
}

/// Formats the UUID the same way as `clinfo` does, as an example:
/// the output should looks like 46abccd6-022e-b783-572d-833f7104d05f
impl fmt::Display for DeviceUuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}-{}-{}-{}-{}",
            hex::encode(&self.0[..4]),
            hex::encode(&self.0[4..6]),
            hex::encode(&self.0[6..8]),
            hex::encode(&self.0[8..10]),
            hex::encode(&self.0[10..])
        )
    }
}

impl fmt::Debug for DeviceUuid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

/// Unique identifier that can either be a PCI ID or a UUID.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum UniqueId {
    PciId(PciId),
    Uuid(DeviceUuid),
}

/// If the string contains a dash, it's interpreted as UUID, else it's interpreted as PCI ID.
impl TryFrom<&str> for UniqueId {
    type Error = GPUError;

    fn try_from(unique_id: &str) -> GPUResult<Self> {
        Ok(match unique_id.contains('-') {
            true => Self::Uuid(DeviceUuid::try_from(unique_id)?),
            false => Self::PciId(PciId::try_from(unique_id)?),
        })
    }
}

impl fmt::Display for UniqueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PciId(id) => id.fmt(f),
            Self::Uuid(id) => id.fmt(f),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Vendor {
    Amd,
    Nvidia,
}

impl TryFrom<&str> for Vendor {
    type Error = GPUError;

    fn try_from(vendor: &str) -> GPUResult<Self> {
        match vendor {
            AMD_DEVICE_VENDOR_STRING => Ok(Self::Amd),
            AMD_DEVICE_ON_APPLE_VENDOR_STRING => Ok(Self::Amd),
            NVIDIA_DEVICE_VENDOR_STRING => Ok(Self::Nvidia),
            _ => Err(GPUError::UnsupportedVendor(vendor.to_string())),
        }
    }
}

impl TryFrom<u32> for Vendor {
    type Error = GPUError;

    fn try_from(vendor: u32) -> GPUResult<Self> {
        match vendor {
            AMD_DEVICE_VENDOR_ID => Ok(Self::Amd),
            AMD_DEVICE_ON_APPLE_VENDOR_ID => Ok(Self::Amd),
            NVIDIA_DEVICE_VENDOR_ID => Ok(Self::Nvidia),
            _ => Err(GPUError::UnsupportedVendor(format!("0x{:x}", vendor))),
        }
    }
}

impl fmt::Display for Vendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let vendor = match self {
            Self::Amd => AMD_DEVICE_VENDOR_STRING,
            Self::Nvidia => NVIDIA_DEVICE_VENDOR_STRING,
        };
        write!(f, "{}", vendor)
    }
}

/// A Buffer to be used for sending and receiving data to/from the GPU.
pub struct Buffer<T> {
    buffer: opencl3::memory::Buffer<T>,
    /// The number of T-sized elements.
    length: usize,
}

/// OpenCL specific device.
#[derive(Debug, Clone)]
pub struct Device {
    vendor: Vendor,
    name: String,
    /// The total memory of the GPU in bytes.
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
    pub fn is_little_endian(&self) -> GPUResult<bool> {
        self.device
            .endian_little()
            .map_err(|_| GPUError::DeviceInfoNotAvailable(CL_DEVICE_ENDIAN_LITTLE))
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

    /// Returns the best possible unique identifier, a UUID is preferred over a PCI ID.
    pub fn unique_id(&self) -> UniqueId {
        match self.uuid {
            Some(uuid) => UniqueId::Uuid(uuid),
            None => UniqueId::PciId(self.pci_id),
        }
    }

    /// Return all available GPU devices of supported vendors.
    pub fn all() -> Vec<&'static Device> {
        Self::all_iter().collect()
    }

    pub fn by_pci_id(pci_id: PciId) -> GPUResult<&'static Device> {
        Self::all_iter()
            .find(|d| pci_id == d.pci_id)
            .ok_or(GPUError::DeviceNotFound)
    }

    pub fn by_uuid(uuid: DeviceUuid) -> GPUResult<&'static Device> {
        Self::all_iter()
            .find(|d| match d.uuid {
                Some(id) => uuid == id,
                None => false,
            })
            .ok_or(GPUError::DeviceNotFound)
    }

    pub fn by_unique_id(unique_id: UniqueId) -> GPUResult<&'static Device> {
        Self::all_iter()
            .find(|d| unique_id == d.unique_id())
            .ok_or(GPUError::DeviceNotFound)
    }

    fn all_iter() -> impl Iterator<Item = &'static Device> {
        utils::DEVICES.iter()
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
#[allow(broken_intra_doc_links)]
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

    /// Creates a program for a specific device from a compiled OpenCL binary.
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

    /// Creates a new buffer that can be used for input/output with the GPU.
    ///
    /// The `length` is the number of elements to create.
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

    /// Puts data from an existing buffer onto the GPU.
    ///
    /// The `offset` is in number of `T` sized elements, not in their byte size.
    pub fn write_from_buffer<T>(
        &self,
        buffer: &Buffer<T>,
        offset: usize,
        data: &[T],
    ) -> GPUResult<()> {
        assert!(offset + data.len() <= buffer.length, "Buffer is too small");

        let mut buff = buffer
            .buffer
            .create_sub_buffer(CL_MEM_READ_WRITE, offset, data.len())?;

        self.queue
            .enqueue_write_buffer(&mut buff, CL_BLOCKING, 0, data, &[])?;

        Ok(())
    }

    /// Reads data from the GPU into an existing buffer.
    ///
    /// The `offset` is in number of `T` sized elements, not in their byte size.
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
    /// On CUDA it sets the correct contexts and synchronizes the stream before returning.
    /// On OpenCL it's only executing the closure without any other side-effects.
    pub fn run<F, R, E>(&self, fun: F) -> Result<R, E>
    where
        F: FnOnce() -> Result<R, E>,
        E: From<GPUError>,
    {
        fun()
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
        kernel.builder.set_arg(&self.buffer);
    }
}

impl KernelArgument for i32 {
    fn push(&self, kernel: &mut Kernel) {
        kernel.builder.set_arg(self);
    }
}

impl KernelArgument for u32 {
    fn push(&self, kernel: &mut Kernel) {
        kernel.builder.set_arg(self);
    }
}

/// A local buffer.
pub struct LocalBuffer<T> {
    /// The number of T sized elements.
    length: usize,
    _phantom: std::marker::PhantomData<T>,
}
impl<T> LocalBuffer<T> {
    /// Returns a new buffer of the specified `length`.
    pub fn new(length: usize) -> Self {
        LocalBuffer::<T> {
            length,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> KernelArgument for LocalBuffer<T> {
    fn push(&self, kernel: &mut Kernel) {
        kernel
            .builder
            .set_arg_local_buffer::<T>(self.length * std::mem::size_of::<T>());
    }
}

/// A kernel that can be executed.
#[derive(Debug)]
pub struct Kernel<'a> {
    builder: ExecuteKernel<'a>,
    queue: &'a CommandQueue,
}

impl<'a> Kernel<'a> {
    /// Set a kernel argument.
    pub fn arg<T: KernelArgument>(mut self, t: &T) -> Self {
        t.push(&mut self);
        self
    }

    /// Actually run the kernel.
    pub fn run(mut self) -> GPUResult<()> {
        self.builder.enqueue_nd_range(&self.queue)?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{
        Device, DeviceUuid, GPUError, PciId, UniqueId, Vendor, AMD_DEVICE_ON_APPLE_VENDOR_ID,
        AMD_DEVICE_ON_APPLE_VENDOR_STRING, AMD_DEVICE_VENDOR_ID, AMD_DEVICE_VENDOR_STRING,
        NVIDIA_DEVICE_VENDOR_ID, NVIDIA_DEVICE_VENDOR_STRING,
    };
    use std::convert::TryFrom;

    #[test]
    fn test_device_all() {
        let devices = Device::all();
        for device in devices.iter() {
            println!("device: {:?}", device);
        }
        assert!(!devices.is_empty(), "No supported GPU found.");
    }

    #[test]
    fn test_vendor_from_str() {
        assert_eq!(
            Vendor::try_from(AMD_DEVICE_VENDOR_STRING).unwrap(),
            Vendor::Amd,
            "AMD vendor string can be converted."
        );
        assert_eq!(
            Vendor::try_from(AMD_DEVICE_ON_APPLE_VENDOR_STRING).unwrap(),
            Vendor::Amd,
            "AMD vendor string (on apple) can be converted."
        );
        assert_eq!(
            Vendor::try_from(NVIDIA_DEVICE_VENDOR_STRING).unwrap(),
            Vendor::Nvidia,
            "Nvidia vendor string can be converted."
        );
        assert!(matches!(
            Vendor::try_from("unknown vendor"),
            Err(GPUError::UnsupportedVendor(_))
        ));
    }

    #[test]
    fn test_vendor_from_u32() {
        assert_eq!(
            Vendor::try_from(AMD_DEVICE_VENDOR_ID).unwrap(),
            Vendor::Amd,
            "AMD vendor ID can be converted."
        );
        assert_eq!(
            Vendor::try_from(AMD_DEVICE_ON_APPLE_VENDOR_ID).unwrap(),
            Vendor::Amd,
            "AMD vendor ID (on apple) can be converted."
        );
        assert_eq!(
            Vendor::try_from(NVIDIA_DEVICE_VENDOR_ID).unwrap(),
            Vendor::Nvidia,
            "Nvidia vendor ID can be converted."
        );
        assert!(matches!(
            Vendor::try_from(0x1abc),
            Err(GPUError::UnsupportedVendor(_))
        ));
    }

    #[test]
    fn test_vendor_display() {
        assert_eq!(
            Vendor::Amd.to_string(),
            AMD_DEVICE_VENDOR_STRING,
            "AMD vendor can be converted to string."
        );
        assert_eq!(
            Vendor::Nvidia.to_string(),
            NVIDIA_DEVICE_VENDOR_STRING,
            "Nvidia vendor can be converted to string."
        );
    }

    #[test]
    fn test_uuid() {
        let valid_string = "46abccd6-022e-b783-572d-833f7104d05f";
        let valid = DeviceUuid::try_from(valid_string).unwrap();
        assert_eq!(valid_string, &valid.to_string());

        let too_short_string = "ccd6-022e-b783-572d-833f7104d05f";
        let too_short = DeviceUuid::try_from(too_short_string);
        assert!(too_short.is_err(), "Parse error when UUID is too short.");

        let invalid_hex_string = "46abccd6-022e-b783-572d-833f7104d05h";
        let invalid_hex = DeviceUuid::try_from(invalid_hex_string);
        assert!(
            invalid_hex.is_err(),
            "Parse error when UUID containts non-hex character."
        );
    }

    #[test]
    fn test_pci_id() {
        let valid_string = "01:00";
        let valid = PciId::try_from(valid_string).unwrap();
        assert_eq!(valid_string, &valid.to_string());
        assert_eq!(valid, PciId(0x0100));

        let too_short_string = "3f";
        let too_short = PciId::try_from(too_short_string);
        assert!(too_short.is_err(), "Parse error when PCI ID is too short.");

        let invalid_hex_string = "aaxx";
        let invalid_hex = PciId::try_from(invalid_hex_string);
        assert!(
            invalid_hex.is_err(),
            "Parse error when PCI ID containts non-hex character."
        );
    }

    #[test]
    fn test_unique_id() {
        let valid_pci_id_string = "aa:bb";
        let valid_pci_id = UniqueId::try_from(valid_pci_id_string).unwrap();
        assert_eq!(valid_pci_id_string, &valid_pci_id.to_string());
        assert_eq!(valid_pci_id, UniqueId::PciId(PciId(0xaabb)));

        let valid_uuid_string = "aabbccdd-eeff-0011-2233-445566778899";
        let valid_uuid = UniqueId::try_from(valid_uuid_string).unwrap();
        assert_eq!(valid_uuid_string, &valid_uuid.to_string());
        assert_eq!(
            valid_uuid,
            UniqueId::Uuid(DeviceUuid([
                0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                0x88, 0x99
            ]))
        );

        let invalid_string = "aabbccddeeffgg";
        let invalid = UniqueId::try_from(invalid_string);
        assert!(
            invalid.is_err(),
            "Parse error when ID matches neither a PCI Id, nor a UUID."
        );
    }
}
