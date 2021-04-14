use std::convert::TryInto;
use std::fmt::Write;

use lazy_static::lazy_static;
use log::{debug, warn};
use sha2::{Digest, Sha256};

use super::{Brand, Device, DeviceUuid, GPUError, GPUResult};

#[repr(C)]
#[derive(Debug, Clone, Default)]
struct cl_amd_device_topology {
    r#type: u32,
    unused: [u8; 17],
    bus: u8,
    device: u8,
    function: u8,
}

const AMD_DEVICE_VENDOR_STRING: &'static str = "AMD";
const NVIDIA_DEVICE_VENDOR_STRING: &'static str = "NVIDIA Corporation";

// constants defined as part of the opencl spec
// https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl_ext.h#L687
const CL_DEVICE_UUID_KHR: u32 = 0x106A;
pub(crate) const CL_UUID_SIZE_KHR: usize = 16;

pub fn is_little_endian(d: ocl::Device) -> GPUResult<bool> {
    match d.info(ocl::enums::DeviceInfo::EndianLittle)? {
        ocl::enums::DeviceInfoResult::EndianLittle(b) => Ok(b),
        _ => Err(GPUError::DeviceInfoNotAvailable(
            ocl::enums::DeviceInfo::EndianLittle,
        )),
    }
}

pub fn get_device_uuid(d: ocl::Device) -> ocl::Result<DeviceUuid> {
    let result = d.info_raw(CL_DEVICE_UUID_KHR)?;
    assert_eq!(result.len(), CL_UUID_SIZE_KHR);
    let mut raw = [0u8; CL_UUID_SIZE_KHR];
    raw.copy_from_slice(result.as_slice());
    Ok(DeviceUuid(raw))
}

pub fn get_pci_id(d: ocl::Device) -> ocl::Result<u32> {
    let vendor = d.vendor()?;
    match vendor.as_str() {
        AMD_DEVICE_VENDOR_STRING => get_amd_pci_id(d),
        NVIDIA_DEVICE_VENDOR_STRING => get_nvidia_pci_id(d),
        _ => Err(ocl::Error::from(format!(
            "cannot get pciId for device with vendor {} ",
            vendor
        ))),
    }
}

fn get_nvidia_pci_id(d: ocl::Device) -> ocl::Result<u32> {
    const CL_DEVICE_PCI_SLOT_ID_NV: u32 = 0x4009;

    let result = d.info_raw(CL_DEVICE_PCI_SLOT_ID_NV)?;
    Ok(u32::from_le_bytes(result[..].try_into().unwrap()))
}

fn get_amd_pci_id(d: ocl::Device) -> ocl::Result<u32> {
    const CL_DEVICE_TOPOLOGY_AMD: u32 = 0x4037;

    let result = d.info_raw(CL_DEVICE_TOPOLOGY_AMD)?;
    let size = std::mem::size_of::<cl_amd_device_topology>();
    assert_eq!(result.len(), size);
    let mut topo = cl_amd_device_topology::default();
    unsafe {
        std::slice::from_raw_parts_mut(&mut topo as *mut cl_amd_device_topology as *mut u8, size)
            .copy_from_slice(&result);
    }
    let device = topo.device as u32;
    let bus = topo.bus as u32;
    let function = topo.function as u32;
    Ok((device << 16) | (bus << 8) | function)
}

pub fn cache_path(device: &Device, cl_source: &str) -> std::io::Result<std::path::PathBuf> {
    let path = dirs::home_dir().unwrap().join(".rust-gpu-tools");
    if !std::path::Path::exists(&path) {
        std::fs::create_dir(&path)?;
    }
    let mut hasher = Sha256::new();
    hasher.input(device.name.as_bytes());
    if let Some(uuid) = device.uuid {
        hasher.input(uuid.to_string());
    }
    if let Some(pci) = device.pci_id {
        hasher.input(pci.to_le_bytes());
    }
    hasher.input(cl_source.as_bytes());
    let mut digest = String::new();
    for &byte in hasher.result()[..].iter() {
        write!(&mut digest, "{:x}", byte).unwrap();
    }
    write!(&mut digest, ".bin").unwrap();

    Ok(path.join(digest))
}

fn get_memory(d: ocl::Device) -> GPUResult<u64> {
    match d.info(ocl::enums::DeviceInfo::GlobalMemSize)? {
        ocl::enums::DeviceInfoResult::GlobalMemSize(sz) => Ok(sz),
        _ => Err(GPUError::DeviceInfoNotAvailable(
            ocl::enums::DeviceInfo::GlobalMemSize,
        )),
    }
}

lazy_static! {
    pub(crate) static ref DEVICES: Vec<Device> = build_device_list();
}

fn build_device_list() -> Vec<Device> {
    let mut all_devices = Vec::new();
    let platforms: Vec<ocl::Platform> = ocl::Platform::list().unwrap_or_default();

    for platform in platforms.iter() {
        let platform_name = match platform.name() {
            Ok(pn) => pn,
            Err(error) => {
                warn!("Cannot get platform name: {:?}", error);
                continue;
            }
        };
        if let Some(brand) = Brand::by_name(&platform_name) {
            let devices = ocl::Device::list(platform, Some(ocl::core::DeviceType::GPU))
                .map_err(Into::into)
                .and_then(|devices| {
                    devices
                        .into_iter()
                        .filter(|d| {
                            if let Ok(vendor) = d.vendor() {
                                match vendor.as_str() {
                                    // Only use devices from the accepted vendors ...
                                    AMD_DEVICE_VENDOR_STRING | NVIDIA_DEVICE_VENDOR_STRING => {
                                        // ... which are available.
                                        return d.is_available().unwrap_or(false);
                                    }
                                    _ => (),
                                }
                            }
                            false
                        })
                        .map(|d| -> GPUResult<_> {
                            Ok(Device {
                                brand,
                                name: d.name()?,
                                memory: get_memory(d)?,
                                uuid: get_device_uuid(d).ok(),
                                pci_id: get_pci_id(d).ok(),
                                platform: *platform,
                                device: d,
                            })
                        })
                        .collect::<GPUResult<Vec<_>>>()
                });
            match devices {
                Ok(mut devices) => {
                    all_devices.append(&mut devices);
                }
                Err(err) => {
                    warn!("Unable to retrieve devices for {:?}: {:?}", brand, err);
                }
            }
        }
    }

    debug!("loaded devices: {:?}", all_devices);
    all_devices
}
