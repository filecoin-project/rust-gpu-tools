use std::convert::{TryFrom, TryInto};
use std::fmt::Write;

use lazy_static::lazy_static;
use log::{debug, warn};
use opencl3::device::DeviceInfo::CL_DEVICE_GLOBAL_MEM_SIZE;
use sha2::{Digest, Sha256};

use super::{Device, GPUError, GPUResult, Vendor};

fn get_bus_id(d: &opencl3::device::Device) -> Result<u32, GPUError> {
    let vendor = Vendor::try_from(d.vendor()?.as_str())?;
    match vendor {
        Vendor::Amd => d.pci_bus_id_amd().map_err(Into::into),
        Vendor::Nvidia => d.pci_bus_id_nv().map_err(Into::into),
    }
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

fn get_memory(d: &opencl3::device::Device) -> GPUResult<u64> {
    d.global_mem_size()
        .map_err(|_| GPUError::DeviceInfoNotAvailable(CL_DEVICE_GLOBAL_MEM_SIZE))
}

lazy_static! {
    pub(crate) static ref DEVICES: Vec<Device> = build_device_list();
}

fn build_device_list() -> Vec<Device> {
    let mut all_devices = Vec::new();
    let platforms: Vec<_> = opencl3::platform::get_platforms().unwrap_or_default();

    for platform in platforms.iter() {
        let devices = platform
            .get_devices(opencl3::device::CL_DEVICE_TYPE_GPU)
            .map_err(Into::into)
            .and_then(|devices| {
                devices
                    .into_iter()
                    .map(opencl3::device::Device::new)
                    .filter(|d| {
                        if let Ok(vendor) = d.vendor() {
                            // Only use devices from the accepted vendors ...
                            if Vendor::try_from(vendor.as_str()).is_ok() {
                                // ... which are available.
                                return d.available().unwrap_or(0) != 0;
                            }
                        }
                        false
                    })
                    .map(|d| -> GPUResult<_> {
                        Ok(Device {
                            vendor: d.vendor()?.as_str().try_into()?,
                            name: d.name()?,
                            memory: get_memory(&d)?,
                            bus_id: get_bus_id(&d).ok(),
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
                let platform_name = platform
                    .name()
                    .unwrap_or_else(|_| "<unknown platform>".to_string());
                warn!(
                    "Unable to retrieve devices for {}: {:?}",
                    platform_name, err
                );
            }
        }
    }

    debug!("loaded devices: {:?}", all_devices);
    all_devices
}
