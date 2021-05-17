use std::convert::TryInto;

use lazy_static::lazy_static;
use log::{debug, warn};

use super::*;

#[repr(C)]
#[derive(Debug, Clone, Default)]
struct cl_amd_device_topology {
    r#type: u32,
    unused: [u8; 17],
    bus: u8,
    device: u8,
    function: u8,
}

const AMD_DEVICE_VENDOR_STRING: &str = "AMD";
const NVIDIA_DEVICE_VENDOR_STRING: &str = "NVIDIA Corporation";

pub fn is_little_endian(d: ocl::Device) -> GPUResult<bool> {
    match d.info(ocl::enums::DeviceInfo::EndianLittle)? {
        ocl::enums::DeviceInfoResult::EndianLittle(b) => Ok(b),
        _ => Err(GPUError::DeviceInfoNotAvailable(
            ocl::enums::DeviceInfo::EndianLittle,
        )),
    }
}

pub fn get_bus_id(d: ocl::Device) -> ocl::Result<u32> {
    let vendor = d.vendor()?;
    match vendor.as_str() {
        AMD_DEVICE_VENDOR_STRING => get_amd_bus_id(d),
        NVIDIA_DEVICE_VENDOR_STRING => get_nvidia_bus_id(d),
        _ => Err(ocl::Error::from(format!(
            "cannot get bus ID for device with vendor {} ",
            vendor
        ))),
    }
}

pub fn get_nvidia_bus_id(d: ocl::Device) -> ocl::Result<u32> {
    const CL_DEVICE_PCI_BUS_ID_NV: u32 = 0x4008;
    let result = d.info_raw(CL_DEVICE_PCI_BUS_ID_NV)?;
    Ok(u32::from_le_bytes(result[..].try_into().unwrap()))
}

pub fn get_amd_bus_id(d: ocl::Device) -> ocl::Result<u32> {
    const CL_DEVICE_TOPOLOGY_AMD: u32 = 0x4037;
    let result = d.info_raw(CL_DEVICE_TOPOLOGY_AMD)?;
    let size = std::mem::size_of::<cl_amd_device_topology>();
    assert_eq!(result.len(), size);
    let mut topo = cl_amd_device_topology::default();
    unsafe {
        std::slice::from_raw_parts_mut(&mut topo as *mut cl_amd_device_topology as *mut u8, size)
            .copy_from_slice(&result);
    }
    Ok(topo.bus as u32)
}

pub fn cache_path(device: &Device, cl_source: &str) -> std::io::Result<std::path::PathBuf> {
    let path = dirs::home_dir().unwrap().join(".rust-gpu-tools");
    if !std::path::Path::exists(&path) {
        std::fs::create_dir(&path)?;
    }
    let mut hasher = Sha256::new();
    // If there are multiple devices with the same name and neither has a Bus-Id,
    // then there will be a collision. Bus-Id can be missing in the case of an Apple
    // GPU. For now, we assume that in the unlikely event of a collision, the same
    // cache can be used.
    // TODO: We might be able to get around this issue by using cl_vendor_id instead of Bus-Id.
    hasher.input(device.name.as_bytes());
    if let Some(bus_id) = device.bus_id {
        hasher.input(bus_id.to_be_bytes());
    }
    hasher.input(cl_source.as_bytes());
    let mut digest = String::new();
    for &byte in hasher.result()[..].iter() {
        write!(&mut digest, "{:x}", byte).unwrap();
    }
    write!(&mut digest, ".bin").unwrap();

    Ok(path.join(digest))
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
                                bus_id: utils::get_bus_id(d).ok(),
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
