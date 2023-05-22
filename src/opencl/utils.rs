use std::convert::TryFrom;

use log::{debug, warn};
use opencl3::device::CL_UUID_SIZE_KHR;
use sha2::{Digest, Sha256};

use crate::device::{DeviceUuid, PciId, Vendor};
use crate::error::{GPUError, GPUResult};
use crate::opencl::Device;

/// The PCI-ID is the combination of the PCI Bus ID and PCI Device ID.
///
/// It is the first two identifiers of e.g. `lspci`:
///
/// ```text
///     4e:00.0 VGA compatible controller
///     || └└-- Device ID
///     └└-- Bus ID
/// ```
fn get_pci_id(device: &opencl3::device::Device) -> GPUResult<PciId> {
    let vendor = Vendor::try_from(device.vendor_id()?)?;
    let id = match vendor {
        Vendor::Amd => {
            let topo = device.topology_amd()?;
            let bus_id = topo.bus as u16;
            let device_id = topo.device as u16;
            (bus_id << 8) | device_id
        }
        Vendor::Intel => {
            let pcibusinfo = device.pcibusinfokhr_intel()?;
            let bus_id = pcibusinfo.pci_bus as u16;
            let device_id = pcibusinfo.pci_device as u16;
            (bus_id << 8) | device_id
        }
        Vendor::Nvidia => {
            let bus_id = device.pci_bus_id_nv()? as u16;
            let device_id = device.pci_slot_id_nv()? as u16;
            (bus_id << 8) | device_id
        }
    };
    Ok(id.into())
}

fn get_uuid(device: &opencl3::device::Device) -> GPUResult<DeviceUuid> {
    let uuid = device.uuid_khr()?;
    Ok(uuid.into())
}

pub fn cache_path(device: &Device, cl_source: &str) -> std::io::Result<std::path::PathBuf> {
    let path = home::home_dir().unwrap().join(".rust-gpu-tools");
    if !std::path::Path::exists(&path) {
        std::fs::create_dir(&path)?;
    }
    let mut hasher = Sha256::new();
    hasher.update(device.name.as_bytes());
    hasher.update(u16::from(device.pci_id).to_be_bytes());
    hasher.update(<[u8; CL_UUID_SIZE_KHR]>::from(
        device.uuid.unwrap_or_default(),
    ));
    hasher.update(cl_source.as_bytes());
    let filename = format!("{}.bin", hex::encode(hasher.finalize()));
    Ok(path.join(filename))
}

fn get_memory(d: &opencl3::device::Device) -> GPUResult<u64> {
    d.global_mem_size()
        .map_err(GPUError::DeviceInfoNotAvailable)
}

fn get_compute_units(d: &opencl3::device::Device) -> GPUResult<u32> {
    d.max_compute_units()
        .map_err(GPUError::DeviceInfoNotAvailable)
}

/// Get the major an minor version of the compute capabilitiy (only available on Nvidia GPUs).
fn get_compute_capability(d: &opencl3::device::Device) -> GPUResult<(u32, u32)> {
    let major = d
        .compute_capability_major_nv()
        .map_err(GPUError::DeviceInfoNotAvailable)?;
    let minor = d
        .compute_capability_major_nv()
        .map_err(GPUError::DeviceInfoNotAvailable)?;
    Ok((major, minor))
}

/// Get a list of all available and supported devices.
///
/// If there is a failure retrieving a device, it won't lead to a hard error, but an error will be
/// logged and the corresponding device won't be available.
pub(crate) fn build_device_list() -> Vec<Device> {
    let mut all_devices = Vec::new();
    let platforms: Vec<_> = opencl3::platform::get_platforms().unwrap_or_default();

    let mut devices_without_pci_id = Vec::new();

    for platform in platforms.iter() {
        let devices = platform
            .get_devices(opencl3::device::CL_DEVICE_TYPE_GPU)
            .map_err(Into::into)
            .and_then(|devices| {
                devices
                    .into_iter()
                    .map(opencl3::device::Device::new)
                    .filter_map(|device| {
                        if let Ok(vendor_id) = device.vendor_id() {
                            // Only use devices from the accepted vendors ...
                            let vendor = Vendor::try_from(vendor_id).ok()?;
                            // ... which are available.
                            if !device.available().unwrap_or(false) {
                                return None;
                            }

                            // `filter_map()` needs to return erros wrapped in an `Option`, hence
                            // early returns with the question mark operator cannot be used.
                            let name = match device.name() {
                                Ok(name) => name,
                                Err(error) => return Some(Err(error.into())),
                            };
                            let memory = match get_memory(&device) {
                                Ok(memory) => memory,
                                Err(error) => return Some(Err(error)),
                            };
                            let compute_units = match get_compute_units(&device) {
                                Ok(units) => units,
                                Err(error) => return Some(Err(error)),
                            };
                            let compute_capability = get_compute_capability(&device).ok();
                            let uuid = get_uuid(&device).ok();

                            // If a device doesn't have a PCI-ID, add those later to the list of
                            // devices with a fake PCI-ID.
                            match get_pci_id(&device) {
                                Ok(pci_id) => {
                                    return Some(Ok(Device {
                                        vendor,
                                        name,
                                        memory,
                                        compute_units,
                                        compute_capability,
                                        pci_id,
                                        uuid,
                                        device,
                                    }));
                                }
                                Err(_) => {
                                    // Use a temporary PCI-ID and replace it later with a
                                    // non-colliding one.
                                    let pci_id = PciId::from(0);
                                    devices_without_pci_id.push(Device {
                                        vendor,
                                        name,
                                        memory,
                                        compute_units,
                                        compute_capability,
                                        pci_id,
                                        uuid,
                                        device,
                                    });
                                    return None;
                                }
                            };
                        }
                        None
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

    // Laptops might have an integrated GPU. Such devices might have neither a PCI-ID, nor a UUID.
    // As those devices are used for development and not for production use, it's good enough to
    // provide a workaround which doesn't add much complexity to the code. We use a fake PCI-ID
    // instead, which is generated by enumerating the available devices. In order to make that
    // case easier to spot when debugging issues, a starting number which is pleasant to the human
    // eye was choosen, that works both, decimal and hexadecimal (4660 == 0x1234).
    let mut enumerated_device: u16 = 4660;
    for mut device in devices_without_pci_id.into_iter() {
        // Make sure that no device has that actual PCI-ID
        while all_devices
            .iter()
            .any(|d| d.pci_id() == enumerated_device.into())
        {
            enumerated_device += 1;
        }
        device.pci_id = PciId::from(enumerated_device);
        enumerated_device += 1;
        all_devices.push(device);
    }

    debug!("loaded devices: {:?}", all_devices);
    all_devices
}
