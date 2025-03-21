//! Utility functions for the Metal implementation.

use log::{debug, error, warn};
use metal::{Device as MetalDevice, MTLGPUFamily};

use crate::device::{PciId, Vendor};
use crate::error::GPUError;
use crate::metal::Device;

/// Build a list of all available Metal devices.
pub fn build_device_list() -> Vec<Device> {
    debug!("Getting Metal devices");
    
    let mut devices = Vec::new();
    
    // Get all available Metal devices
    let all_devices = MetalDevice::all();
    
    for metal_device in all_devices {
        match build_device(&metal_device) {
            Ok(device) => devices.push(device),
            Err(e) => error!("Error building Metal device: {}", e),
        }
    }
    
    devices
}

/// Builds a Metal device from a MetalDevice.
fn build_device(metal_device: &MetalDevice) -> Result<Device, GPUError> {
    let name = metal_device.name().to_string();
    debug!("Found Metal device: {}", name);
    
    // Determine vendor
    let vendor = if metal_device.supports_family(MTLGPUFamily::Apple1) ||
                  metal_device.supports_family(MTLGPUFamily::Apple2) ||
                  metal_device.supports_family(MTLGPUFamily::Apple3) ||
                  metal_device.supports_family(MTLGPUFamily::Apple4) ||
                  metal_device.supports_family(MTLGPUFamily::Apple5) ||
                  metal_device.supports_family(MTLGPUFamily::Apple6) ||
                  metal_device.supports_family(MTLGPUFamily::Apple7) ||
                  metal_device.supports_family(MTLGPUFamily::Apple8) {
        Vendor::Apple
    } else if name.to_lowercase().contains("amd") || name.to_lowercase().contains("radeon") {
        Vendor::Amd
    } else if name.to_lowercase().contains("intel") || name.to_lowercase().contains("iris") {
        Vendor::Intel
    } else {
        warn!("Unknown Metal device vendor: {}, assuming Apple", name);
        Vendor::Apple
    };
    
    // Create a pseudo PCI ID for Metal devices since Metal doesn't expose PCI IDs
    // Use a hash of the device name to create a consistent ID
    let pci_id_value = name.bytes().fold(0u16, |acc, b| acc.wrapping_add(b as u16));
    let pci_id = PciId::from(pci_id_value);
    
    // Create a UUID if possible (currently not available in Metal, so we use None)
    let uuid = None;
    
    // Get memory size - Metal doesn't directly expose this, so we'll use a large value
    // that still makes the device usable for most workloads
    let memory = metal_device.recommended_max_working_set_size();
    
    // Metal doesn't directly expose compute units, so we create an estimation
    // based on a reasonable assumption for Apple GPUs
    let max_tgs = 1024; // Just use a reasonable number here
    let compute_units = (max_tgs / 1024).max(1) as u32;
    
    Ok(Device {
        vendor,
        name: name.clone(),
        memory,
        compute_units,
        pci_id,
        uuid,
        device_name: name,
    })
}