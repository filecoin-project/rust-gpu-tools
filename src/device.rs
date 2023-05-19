use std::fmt;

use log::debug;
#[cfg(all(feature = "opencl", feature = "cuda"))]
use log::warn;
use once_cell::sync::Lazy;

use std::convert::TryFrom;
use std::mem;

use crate::error::{GPUError, GPUResult};

#[cfg(feature = "cuda")]
use crate::cuda;
#[cfg(feature = "opencl")]
use crate::opencl;

/// The UUID of the devices returned by OpenCL as well as CUDA are always 16 bytes long.
const UUID_SIZE: usize = 16;
const AMD_DEVICE_VENDOR_STRING: &str = "Advanced Micro Devices, Inc.";
const AMD_DEVICE_VENDOR_ID: u32 = 0x1002;

const INTEL_DEVICE_VENDOR_STRING: &str = "Intel(R) Corporation";
const INTEL_DEVICE_VENDOR_ID: u32 = 0x8086;

// For some reason integrated AMD cards on Apple don't have the usual vendor name and ID
const AMD_DEVICE_ON_APPLE_VENDOR_STRING: &str = "AMD";
const AMD_DEVICE_ON_APPLE_VENDOR_ID: u32 = 0x1021d00;
const NVIDIA_DEVICE_VENDOR_STRING: &str = "NVIDIA Corporation";
const NVIDIA_DEVICE_VENDOR_ID: u32 = 0x10de;

// The owned CUDA contexts are stored globally. Each devives contains an unowned reference, so
// that devices can be cloned.
#[cfg(feature = "cuda")]
static DEVICES: Lazy<(Vec<Device>, cuda::utils::CudaContexts)> = Lazy::new(build_device_list);

// Keep it as a tuple as the CUDA case, so that the using `DEVICES` is independent of the
// features set.
#[cfg(all(feature = "opencl", not(feature = "cuda")))]
static DEVICES: Lazy<(Vec<Device>, ())> = Lazy::new(build_device_list);

/// The PCI-ID is the combination of the PCI Bus ID and PCI Device ID.
///
/// It is the first two identifiers of e.g. `lcpci`:
///
/// ```text
///     4e:00.0 VGA compatible controller
///     || └└-- Device ID
///     └└-- Bus ID
/// ```
#[derive(Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
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
        hex::decode_to_slice(pci_id.replace(':', ""), &mut bytes).map_err(|_| {
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
#[derive(Copy, Clone, Default, Eq, Hash, PartialEq)]
pub struct DeviceUuid([u8; UUID_SIZE]);

impl From<[u8; UUID_SIZE]> for DeviceUuid {
    fn from(uuid: [u8; UUID_SIZE]) -> Self {
        Self(uuid)
    }
}

impl From<DeviceUuid> for [u8; UUID_SIZE] {
    fn from(uuid: DeviceUuid) -> Self {
        uuid.0
    }
}

/// Converts a UUID formatted as aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee,
/// e.g. 46abccd6-022e-b783-572d-833f7104d05f
impl TryFrom<&str> for DeviceUuid {
    type Error = GPUError;

    fn try_from(uuid: &str) -> GPUResult<Self> {
        let mut bytes = [0; UUID_SIZE];
        hex::decode_to_slice(uuid.replace('-', ""), &mut bytes)
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
        write!(f, "{}", self)
    }
}

/// Unique identifier that can either be a PCI ID or a UUID.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum UniqueId {
    /// ID based on the PCI bus.
    PciId(PciId),
    /// ID based on a globally unique identifier.
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

/// Currently supported vendors of this library.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Vendor {
    /// GPU by AMD.
    Amd,
    /// GPU by Intel.
    Intel,
    /// GPU by NVIDIA.
    Nvidia,
}

impl TryFrom<&str> for Vendor {
    type Error = GPUError;

    fn try_from(vendor: &str) -> GPUResult<Self> {
        match vendor {
            AMD_DEVICE_VENDOR_STRING => Ok(Self::Amd),
            AMD_DEVICE_ON_APPLE_VENDOR_STRING => Ok(Self::Amd),
            INTEL_DEVICE_VENDOR_STRING => Ok(Self::Intel),
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
            INTEL_DEVICE_VENDOR_ID => Ok(Self::Intel),
            NVIDIA_DEVICE_VENDOR_ID => Ok(Self::Nvidia),
            _ => Err(GPUError::UnsupportedVendor(format!("0x{:x}", vendor))),
        }
    }
}

impl fmt::Display for Vendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let vendor = match self {
            Self::Amd => AMD_DEVICE_VENDOR_STRING,
            Self::Intel => INTEL_DEVICE_VENDOR_STRING,
            Self::Nvidia => NVIDIA_DEVICE_VENDOR_STRING,
        };
        write!(f, "{}", vendor)
    }
}

/// Which framework to use, CUDA or OpenCL.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Framework {
    /// CUDA.
    #[cfg(feature = "cuda")]
    Cuda,
    /// OpenCL.
    #[cfg(feature = "opencl")]
    Opencl,
}

/// A device that may have a CUDA and/or OpenCL GPU associated with it.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Device {
    vendor: Vendor,
    name: String,
    memory: u64,
    compute_units: u32,
    /// Major and minor version of the compute capabilitiy (only available on Nvidia GPUs).
    compute_capability: Option<(u32, u32)>,
    // All devices have a PCI ID. It is used as fallback in case there is not UUID.
    pci_id: PciId,
    uuid: Option<DeviceUuid>,
    #[cfg(feature = "cuda")]
    cuda: Option<cuda::Device>,
    #[cfg(feature = "opencl")]
    opencl: Option<opencl::Device>,
}

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

    /// Returns the best possible unique identifier, a UUID is preferred over a PCI ID.
    pub fn unique_id(&self) -> UniqueId {
        match self.uuid {
            Some(uuid) => UniqueId::Uuid(uuid),
            None => UniqueId::PciId(self.pci_id),
        }
    }

    /// Returns the preferred framework (CUDA or OpenCL) to use.
    ///
    /// CUDA will be be preferred over OpenCL. The returned framework will work on the device.
    /// E.g. it won't return `Framework::Cuda` for an AMD device.
    pub fn framework(&self) -> Framework {
        #[cfg(all(feature = "opencl", feature = "cuda"))]
        if cfg!(feature = "cuda") && self.cuda.is_some() {
            Framework::Cuda
        } else {
            Framework::Opencl
        }

        #[cfg(all(feature = "cuda", not(feature = "opencl")))]
        {
            Framework::Cuda
        }

        #[cfg(all(feature = "opencl", not(feature = "cuda")))]
        {
            Framework::Opencl
        }
    }

    /// Returns the underlying CUDA device if it is available.
    #[cfg(feature = "cuda")]
    pub fn cuda_device(&self) -> Option<&cuda::Device> {
        self.cuda.as_ref()
    }

    /// Returns the underlying OpenCL device if it is available.
    #[cfg(feature = "opencl")]
    pub fn opencl_device(&self) -> Option<&opencl::Device> {
        self.opencl.as_ref()
    }

    /// Returns all available GPUs that are supported.
    pub fn all() -> Vec<&'static Device> {
        Self::all_iter().collect()
    }

    /// Returns the device matching the PCI ID if there is one.
    pub fn by_pci_id(pci_id: PciId) -> Option<&'static Device> {
        Self::all_iter().find(|d| pci_id == d.pci_id)
    }

    /// Returns the device matching the UUID if there is one.
    pub fn by_uuid(uuid: DeviceUuid) -> Option<&'static Device> {
        Self::all_iter().find(|d| Some(uuid) == d.uuid)
    }

    /// Returns the device matching the unique ID if there is one.
    pub fn by_unique_id(unique_id: UniqueId) -> Option<&'static Device> {
        Self::all_iter().find(|d| unique_id == d.unique_id())
    }

    /// Returns the device matching the Vendor.
    pub fn by_vendor(vendor_id: Vendor) -> Vec<&'static Device> {
        Self::all_iter()
            .filter(|d| vendor_id == d.vendor())
            .collect()
    }

    /// Returns an iterator of all available GPUs that are supported.
    fn all_iter() -> impl Iterator<Item = &'static Device> {
        DEVICES.0.iter()
    }
}

/// Get a list of all available and supported devices.
///
/// If both, the `cuda` and the `opencl` feature are enabled, a device supporting both will be
/// combined into a single device. You can then access the underlying CUDA and OpenCL device
/// if needed.
///
/// If there is a failure retrieving a device, it won't lead to a hard error, but an error will be
/// logged and the corresponding device won't be available.
#[cfg(feature = "cuda")]
fn build_device_list() -> (Vec<Device>, cuda::utils::CudaContexts) {
    let mut all_devices = Vec::new();

    #[cfg(feature = "opencl")]
    let opencl_devices = opencl::utils::build_device_list();

    #[cfg(all(feature = "cuda", feature = "opencl"))]
    let (mut cuda_devices, cuda_contexts) = cuda::utils::build_device_list();
    #[cfg(all(feature = "cuda", not(feature = "opencl")))]
    let (cuda_devices, cuda_contexts) = cuda::utils::build_device_list();

    // Combine OpenCL and CUDA devices into one device if it is the same GPU
    #[cfg(feature = "opencl")]
    for opencl_device in opencl_devices {
        let mut device = Device {
            vendor: opencl_device.vendor(),
            name: opencl_device.name(),
            memory: opencl_device.memory(),
            compute_units: opencl_device.compute_units(),
            compute_capability: opencl_device.compute_capability(),
            pci_id: opencl_device.pci_id(),
            uuid: opencl_device.uuid(),
            opencl: Some(opencl_device),
            cuda: None,
        };

        // Only devices from Nvidia can use CUDA
        #[cfg(feature = "cuda")]
        if device.vendor == Vendor::Nvidia {
            for ii in 0..cuda_devices.len() {
                if (device.uuid.is_some() && cuda_devices[ii].uuid() == device.uuid)
                    || (cuda_devices[ii].pci_id() == device.pci_id)
                {
                    if device.memory() != cuda_devices[ii].memory() {
                        warn!("OpenCL and CUDA report different amounts of memory for a device with the same identifier");
                        break;
                    }
                    if device.compute_units() != cuda_devices[ii].compute_units() {
                        warn!("OpenCL and CUDA report different amounts of compute units for a device with the same identifier");
                        break;
                    }
                    // Move the CUDA device out of the vector
                    device.cuda = Some(cuda_devices.remove(ii));
                    // Only one device can match
                    break;
                }
            }
        }

        all_devices.push(device)
    }

    // All CUDA devices that don't have a corresponding OpenCL devices
    for cuda_device in cuda_devices {
        let device = Device {
            vendor: cuda_device.vendor(),
            name: cuda_device.name(),
            memory: cuda_device.memory(),
            compute_units: cuda_device.compute_units(),
            compute_capability: Some(cuda_device.compute_capability()),
            pci_id: cuda_device.pci_id(),
            uuid: cuda_device.uuid(),
            cuda: Some(cuda_device),
            #[cfg(feature = "opencl")]
            opencl: None,
        };
        all_devices.push(device);
    }

    debug!("loaded devices: {:?}", all_devices);
    (all_devices, cuda_contexts)
}

/// Get a list of all available and supported OpenCL devices.
///
/// If there is a failure retrieving a device, it won't lead to a hard error, but an error will be
/// logged and the corresponding device won't be available.
#[cfg(all(feature = "opencl", not(feature = "cuda")))]
fn build_device_list() -> (Vec<Device>, ()) {
    let devices = opencl::utils::build_device_list()
        .into_iter()
        .map(|device| Device {
            vendor: device.vendor(),
            name: device.name(),
            memory: device.memory(),
            compute_units: device.compute_units(),
            compute_capability: device.compute_capability(),
            pci_id: device.pci_id(),
            uuid: device.uuid(),
            opencl: Some(device),
        })
        .collect();

    debug!("loaded devices: {:?}", devices);
    (devices, ())
}

#[cfg(test)]
mod test {
    use super::{
        Device, DeviceUuid, GPUError, PciId, UniqueId, Vendor, AMD_DEVICE_ON_APPLE_VENDOR_ID,
        AMD_DEVICE_ON_APPLE_VENDOR_STRING, AMD_DEVICE_VENDOR_ID, AMD_DEVICE_VENDOR_STRING,
        INTEL_DEVICE_VENDOR_ID, INTEL_DEVICE_VENDOR_STRING, NVIDIA_DEVICE_VENDOR_ID,
        NVIDIA_DEVICE_VENDOR_STRING,
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
            Vendor::try_from(INTEL_DEVICE_VENDOR_STRING).unwrap(),
            Vendor::Intel,
            "Intel vendor string can be converted."
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
            Vendor::try_from(INTEL_DEVICE_VENDOR_ID).unwrap(),
            Vendor::Intel,
            "Intel vendor ID can be converted."
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
            Vendor::Intel.to_string(),
            INTEL_DEVICE_VENDOR_STRING,
            "Intel vendor can be converted to string."
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
