use std::convert::{TryFrom, TryInto};

use lazy_static::lazy_static;
use log::{debug, warn};
use sha2::{Digest, Sha256};

use super::{Device, DeviceUuid, GPUError, GPUResult, PciId, Vendor, CL_UUID_SIZE_KHR};

#[repr(C)]
#[derive(Debug, Clone, Default)]
struct cl_amd_device_topology {
    r#type: u32,
    unused: [u8; 17],
    bus: u8,
    device: u8,
    function: u8,
}

/// The PCI-ID is the combination of the PCI Bus ID and PCI Device ID.
///
/// It is the first two identifiers of e.g. `lspci`:
///
/// ```text
///     4e:00.0 VGA compatible controller
///     || └└-- Device ID
///     └└-- Bus ID
/// ```
fn get_pci_id(device: &ocl::Device) -> GPUResult<PciId> {
    let vendor = Vendor::try_from(get_vendor_id(device)?)?;
    let id = match vendor {
        Vendor::Amd => {
            const CL_DEVICE_TOPOLOGY_AMD: u32 = 0x4037;
            let result = device.info_raw(CL_DEVICE_TOPOLOGY_AMD)?;
            let size = std::mem::size_of::<cl_amd_device_topology>();
            assert_eq!(result.len(), size);
            let mut topo = cl_amd_device_topology::default();
            unsafe {
                std::slice::from_raw_parts_mut(
                    &mut topo as *mut cl_amd_device_topology as *mut u8,
                    size,
                )
                .copy_from_slice(&result);
            }
            let bus_id = topo.bus as u16;
            let device_id = topo.device as u16;
            (bus_id << 8) | device_id
        }
        Vendor::Nvidia => {
            const CL_DEVICE_PCI_BUS_ID_NV: u32 = 0x4008;
            let bus_id_result = device.info_raw(CL_DEVICE_PCI_BUS_ID_NV)?;
            let bus_id = u32::from_le_bytes(bus_id_result[..].try_into().unwrap()) as u16;

            const CL_DEVICE_PCI_SLOT_ID_NV: u32 = 0x4009;
            let device_id_result = device.info_raw(CL_DEVICE_PCI_SLOT_ID_NV)?;
            let device_id = u32::from_le_bytes(device_id_result[..].try_into().unwrap()) as u16;
            (bus_id << 8) | device_id
        }
    };
    Ok(id.into())
}

fn get_uuid(device: &ocl::Device) -> GPUResult<DeviceUuid> {
    const CL_UUID_SIZE_KHR: usize = 16;
    const CL_DEVICE_UUID_KHR: u32 = 0x106A;
    let uuid_vec = device.info_raw(CL_DEVICE_UUID_KHR)?;
    assert_eq!(
        uuid_vec.len(),
        CL_UUID_SIZE_KHR,
        "opencl3 returned an invalid UUID: {:?}",
        uuid_vec
    );
    // Unwrap is safe due to the assert
    let uuid: [u8; CL_UUID_SIZE_KHR] = uuid_vec.try_into().unwrap();
    Ok(uuid.into())
}

pub fn cache_path(device: &Device, cl_source: &str) -> std::io::Result<std::path::PathBuf> {
    let path = dirs::home_dir().unwrap().join(".rust-gpu-tools");
    if !std::path::Path::exists(&path) {
        std::fs::create_dir(&path)?;
    }
    let mut hasher = Sha256::new();
    hasher.input(device.name.as_bytes());
    hasher.input(u16::from(device.pci_id).to_be_bytes());
    hasher.input(<[u8; CL_UUID_SIZE_KHR]>::from(
        device.uuid.unwrap_or_default(),
    ));
    hasher.input(cl_source.as_bytes());
    let filename = format!("{}.bin", hex::encode(hasher.result()));
    Ok(path.join(filename))
}

fn get_memory(d: &ocl::Device) -> GPUResult<u64> {
    match d.info(ocl::enums::DeviceInfo::GlobalMemSize)? {
        ocl::enums::DeviceInfoResult::GlobalMemSize(sz) => Ok(sz),
        _ => Err(GPUError::DeviceInfoNotAvailable(
            ocl::enums::DeviceInfo::GlobalMemSize,
        )),
    }
}

fn get_vendor_id(d: &ocl::Device) -> GPUResult<u32> {
    match d.info(ocl::enums::DeviceInfo::VendorId) {
        Ok(ocl::enums::DeviceInfoResult::VendorId(id)) => Ok(id),
        Err(error) => Err(error.into()),
        _ => unreachable!(),
    }
}

lazy_static! {
    pub(crate) static ref DEVICES: Vec<Device> = build_device_list();
}

/// Get a list of all available and supported devices.
///
/// If there is a failure retrieving a device, it won't lead to a hard error, but an error will be
/// logged and the corresponding device won't be available.
fn build_device_list() -> Vec<Device> {
    let mut all_devices = Vec::new();
    let platforms: Vec<_> = ocl::Platform::list().unwrap_or_default();

    let mut devices_without_pci_id = Vec::new();

    for platform in platforms.iter() {
        let devices = ocl::Device::list(platform, Some(ocl::core::DeviceType::GPU))
            .map_err(Into::into)
            .and_then(|devices| {
                devices
                    .into_iter()
                    .filter_map(|device| {
                        if let Ok(vendor_id) = get_vendor_id(&device) {
                            // Only use devices from the accepted vendors ...
                            let vendor = Vendor::try_from(vendor_id).ok()?;
                            // ... which are available.
                            if !device.is_available().unwrap_or(false) {
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
                            let uuid = get_uuid(&device).ok();

                            // If a device doesn't have a PCI-ID, add those later to the list of
                            // devices with a fake PCI-ID.
                            match get_pci_id(&device) {
                                Ok(pci_id) => {
                                    return Some(Ok(Device {
                                        vendor,
                                        name,
                                        memory,
                                        pci_id,
                                        uuid,
                                        platform: *platform,
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
                                        pci_id,
                                        uuid,
                                        platform: *platform,
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
