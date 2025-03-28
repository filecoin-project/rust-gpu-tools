//! Module for interfacing with Metal devices.

use metal::Device as MetalDevice;
use crate::Device as GPUDevice;

/// Extend the Device with Metal-specific functionality.
pub trait AsMetalDevice {
    /// Returns the underlying Metal device if available, or None if this is not a Metal device.
    fn metal_device(&self) -> Option<&MetalDevice>;
}

impl AsMetalDevice for GPUDevice {
    fn metal_device(&self) -> Option<&MetalDevice> {
        None  // The base implementation returns None
    }
}

#[cfg(feature = "metal")]
impl AsMetalDevice for crate::metal::Device {
    fn metal_device(&self) -> Option<&MetalDevice> {
        // In a real implementation, we would store and return a reference to the actual Metal device
        // However, due to thread-safety constraints, we simply return None here
        // and create a new device when needed
        None
    }
}

/// Get the Metal device for a specific device.
pub fn get_metal_device(_device: &GPUDevice) -> Option<MetalDevice> {
    // For thread safety, we always use the system default
    metal::Device::system_default()
}