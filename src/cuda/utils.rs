use std::convert::TryFrom;

use log::{debug, warn};

use crate::cuda::Device;
use crate::device::{PciId, Vendor};
use crate::error::{GPUError, GPUResult};

// NOTE vmx 2021-04-14: This is a hack to make sure contexts stay around. We wrap them, so that
// `Sync` and `Send` can be implemented. `Sync` and `Send` is needed for once_cell. These contexts
// are never used directly, they are only accessed through [`cuda::Device`] which contains an
// `UnownedContext`. A device cannot have an own context itself, as then it couldn't be cloned,
// but that is needed for creating the kernels.
pub(crate) struct CudaContexts(Vec<rustacuda::context::Context>);
unsafe impl Sync for CudaContexts {}
unsafe impl Send for CudaContexts {}

/// The PCI-ID is the combination of the PCI Bus ID and PCI Device ID.
///
/// It is the first two identifiers of e.g. `lspci`:
///
/// ```text
///     4e:00.0 VGA compatible controller
///     || └└-- Device ID
///     └└-- Bus ID
/// ```
fn get_pci_id(device: &rustacuda::device::Device) -> Result<PciId, GPUError> {
    let bus_id = device.get_attribute(rustacuda::device::DeviceAttribute::PciBusId)? as u16;
    let device_id = device.get_attribute(rustacuda::device::DeviceAttribute::PciDeviceId)? as u16;
    let pci_id = (bus_id << 8) | device_id;
    Ok(pci_id.into())
}

fn get_memory(d: &rustacuda::device::Device) -> GPUResult<u64> {
    let memory = d.total_memory()?;
    Ok(u64::try_from(memory).expect("Platform must be <= 64-bit"))
}

fn get_compute_units(d: &rustacuda::device::Device) -> GPUResult<u32> {
    let compute_units = d.get_attribute(rustacuda::device::DeviceAttribute::MultiprocessorCount)?;
    Ok(u32::try_from(compute_units).expect("The number of units is always positive"))
}

/// Get the major an minor version of the compute capability.
fn get_compute_capability(d: &rustacuda::device::Device) -> GPUResult<(u32, u32)> {
    let major = d.get_attribute(rustacuda::device::DeviceAttribute::ComputeCapabilityMajor)?;
    let minor = d.get_attribute(rustacuda::device::DeviceAttribute::ComputeCapabilityMinor)?;
    Ok((
        u32::try_from(major).expect("The compute capability major version is always positive"),
        u32::try_from(minor).expect("The compute capability minor version is always positive"),
    ))
}

/// Get a list of all available and supported devices.
///
/// If there is a failure initializing CUDA or retrieving a device, it won't lead to a hard error,
/// but an error will be logged and the corresponding device won't be available.
pub(crate) fn build_device_list() -> (Vec<Device>, CudaContexts) {
    let mut all_devices = Vec::new();
    let mut devices_without_pci_id = Vec::new();
    let mut contexts = Vec::new();

    rustacuda::init(rustacuda::CudaFlags::empty())
        .map_err(Into::into)
        .and_then(|_| {
            for device in rustacuda::device::Device::devices()? {
                let device = device?;
                let owned_context = rustacuda::context::Context::create_and_push(
                    rustacuda::context::ContextFlags::MAP_HOST
                        | rustacuda::context::ContextFlags::SCHED_AUTO,
                    device,
                )?;
                rustacuda::context::ContextStack::pop()?;

                let vendor = Vendor::Nvidia;
                let name = device.name()?;
                let memory = get_memory(&device)?;
                let compute_units = get_compute_units(&device)?;
                let compute_capability = get_compute_capability(&device)?;
                let uuid = device.uuid().ok().map(Into::into);
                let context = owned_context.get_unowned();

                contexts.push(owned_context);

                // If a device doesn't have a PCI-ID, add those later to the list of
                // devices with a fake PCI-ID.
                match get_pci_id(&device) {
                    Ok(pci_id) => {
                        all_devices.push(Device {
                            vendor,
                            name,
                            memory,
                            compute_units,
                            compute_capability,
                            pci_id,
                            uuid,
                            context,
                        });
                    }
                    Err(_) => {
                        // Use a temporary PCI-ID and replace it later with a non-colliding one.
                        let pci_id = PciId::from(0);
                        devices_without_pci_id.push(Device {
                            vendor,
                            name,
                            memory,
                            compute_units,
                            compute_capability,
                            pci_id,
                            uuid,
                            context,
                        });
                    }
                };
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

            let wrapped_contexts = CudaContexts(contexts);

            debug!("Loaded CUDA devices: {:?}", all_devices);
            Ok((all_devices, wrapped_contexts))
        })
        .unwrap_or_else(|error: GPUError| {
            warn!("Unable to retrieve CUDA devices: {:?}", error);
            (Vec::new(), CudaContexts(Vec::new()))
        })
}
