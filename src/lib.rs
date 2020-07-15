use sha2::{Digest, Sha256};
use std::fmt::Write;

pub type BusId = u64;

fn cache_path(
    device_name: &str,
    bus_id: BusId,
    cl_source: &str,
) -> std::io::Result<std::path::PathBuf> {
    let path = dirs::home_dir().unwrap().join(".rust-gpu-tools");
    if !std::path::Path::exists(&path) {
        std::fs::create_dir(&path)?;
    }
    let mut hasher = Sha256::new();
    hasher.input(device_name.as_bytes());
    hasher.input(bus_id.to_be_bytes());
    hasher.input(cl_source.as_bytes());
    let mut digest = String::new();
    for &byte in hasher.result()[..].iter() {
        write!(&mut digest, "{:x}", byte).unwrap();
    }
    write!(&mut digest, ".bin").unwrap();

    Ok(path.join(digest))
}

fn find_platform(platform_name: &str) -> Option<ocl::Platform> {
    ocl::Platform::list()
        .unwrap()
        .into_iter()
        .find(|&p| match p.name() {
            Ok(p) => p == platform_name.to_string(),
            Err(_) => false,
        })
}

#[derive(Debug, Clone, Copy)]
pub enum Brand {
    Amd,
    Nvidia,
}

#[derive(Debug, Clone, Copy)]
pub struct Device {
    brand: Brand,
    bus_id: BusId,
    device: ocl::Device,
}

impl Brand {
    fn extract_bus_id(&self, d: ocl::Device) -> BusId {
        match self {
            Brand::Nvidia => unimplemented!(),
            Brand::Amd => unimplemented!(),
        }
    }
    pub fn get_devices(&self) -> Vec<Device> {
        ocl::Device::list_all(
            find_platform(match self {
                Brand::Nvidia => "NVIDIA CUDA",
                Brand::Amd => "AMD Accelerated Parallel Processing",
            })
            .unwrap(),
        )
        .unwrap()
        .into_iter()
        .map(|d| Device {
            brand: *self,
            bus_id: self.extract_bus_id(d),
            device: d,
        })
        .collect()
    }
}

pub struct Program {
    pub proque__: ocl::ProQue,
}

impl Program {
    pub fn from_opencl(device: Device, src: &str) -> Program {
        let cached = cache_path(&device.device.name().unwrap(), device.bus_id, src).unwrap();
        if std::path::Path::exists(&cached) {
            let bin = std::fs::read(cached).unwrap();
            Program::from_binary(device, bin)
        } else {
            let prog = Program {
                proque__: ocl::ProQue::builder()
                    .device(device.device)
                    .src(src)
                    .dims(1)
                    .build()
                    .unwrap(),
            };
            std::fs::write(cached, prog.to_binary());
            prog
        }
    }
    pub fn from_binary(device: Device, bin: Vec<u8>) -> Program {
        let bins = vec![&bin[..]];
        let mut prog_builder = ocl::builders::ProgramBuilder::new();
        prog_builder.binaries(&bins);
        Program {
            proque__: ocl::ProQue::builder()
                .device(device.device)
                .prog_bldr(prog_builder)
                .dims(1)
                .build()
                .unwrap(),
        }
    }
    pub fn to_binary(&self) -> Vec<u8> {
        match self
            .proque__
            .program()
            .info(ocl::enums::ProgramInfo::Binaries)
            .unwrap()
        {
            ocl::enums::ProgramInfoResult::Binaries(bins) => {
                return bins[0].clone();
            }
            _ => panic!(),
        }
    }
}

#[macro_export]
macro_rules! call_kernel {
    ($program:expr, $name:expr, $gws:expr, $($arg:expr),*) => {{
        let kernel =
            $program
            .proque__
            .kernel_builder($name)
            .global_work_size($gws)
            $(.arg($arg))*
            .build().unwrap();
        unsafe {
            kernel.enq().unwrap();
        }
    }};
}
