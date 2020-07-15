pub type BusId = u64;

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
        Program {
            proque__: ocl::ProQue::builder()
                .device(device.device)
                .src(src)
                .dims(1)
                .build()
                .unwrap(),
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
