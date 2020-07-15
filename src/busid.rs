use std::convert::TryInto;

pub fn get_nvidia_bus_id(d: ocl::Device) -> ocl::Result<u32> {
    const CL_DEVICE_PCI_BUS_ID_NV: u32 = 0x4008;
    let result = d.info_raw(CL_DEVICE_PCI_BUS_ID_NV)?;
    Ok(u32::from_le_bytes(result[..].try_into().unwrap()))
}
