//! Module for interfacing with Metal devices.

use std::sync::Mutex;
use once_cell::sync::Lazy;
use log::debug;
use metal::Device as MetalDevice;
use crate::Device as GPUDevice;

/// Global device lock for ensuring thread safety when interacting with Metal devices
static METAL_DEVICE_LOCK: Lazy<Mutex<()>> = Lazy::new(|| {
    debug!("Initialized Metal device lock");
    Mutex::new(())
});

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
        self.metal_context.as_ref()
    }
}

/// Get the Metal device for a specific device.
pub fn get_metal_device(device: &GPUDevice) -> Option<MetalDevice> {
    if let Some(metal_device) = device.metal_device() {
        if let Some(context) = metal_device.metal_device() {
            return Some(context.clone());
        }
    }
    
    // If no specific device is found, return None
    None
}

/// Acquire a lock when doing operations that require thread safety with Metal devices
pub fn acquire_metal_lock() -> std::sync::MutexGuard<'static, ()> {
    METAL_DEVICE_LOCK.lock().expect("Metal device lock was poisoned")
}