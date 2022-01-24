use std::collections::HashMap;
use std::env;

use log::info;
use once_cell::sync::Lazy;

/// The number of CUDA cores.
///
/// For non CUDA cards, a number is estimated based on the OpenCL compute units is chosen.
pub static CUDA_CORES: Lazy<HashMap<String, usize>> = Lazy::new(cuda_cores);

fn cuda_cores() -> HashMap<String, usize> {
    let mut core_counts: HashMap<String, usize> = vec![
        // AMD
        ("gfx1010".to_string(), 2560),
        // This value was chosen to give (approximately) empirically best performance for a Radeon Pro VII.
        ("gfx906".to_string(), 7400),
        // NVIDIA
        ("Quadro RTX 6000".to_string(), 4608),
        ("Quadro RTX A6000".to_string(), 10752),
        ("TITAN RTX".to_string(), 4608),
        ("Tesla V100".to_string(), 5120),
        ("Tesla P100".to_string(), 3584),
        ("Tesla T4".to_string(), 2560),
        ("Quadro M5000".to_string(), 2048),
        ("GeForce RTX 3090".to_string(), 10496),
        ("GeForce RTX 3080".to_string(), 8704),
        ("GeForce RTX 3080 Ti".to_string(), 10240),
        ("GeForce RTX 3070".to_string(), 5888),
        ("GeForce RTX 2080 Ti".to_string(), 4352),
        ("GeForce RTX 2080 SUPER".to_string(), 3072),
        ("GeForce RTX 2080".to_string(), 2944),
        ("GeForce RTX 2070 SUPER".to_string(), 2560),
        ("GeForce GTX 1080 Ti".to_string(), 3584),
        ("GeForce GTX 1080".to_string(), 2560),
        ("GeForce GTX 2060".to_string(), 1920),
        ("GeForce GTX 1660 Ti".to_string(), 1536),
        ("GeForce GTX 1060".to_string(), 1280),
        ("GeForce GTX 1650 SUPER".to_string(), 1280),
        ("GeForce GTX 1650".to_string(), 896),
        // CUDA appears to have a different device name
        ("NVIDIA GeForce RTX 3090".to_string(), 10496),
        ("NVIDIA GeForce RTX 3080 Ti".to_string(), 10240),
        ("NVIDIA GeForce RTX 3080".to_string(), 8704),
        ("NVIDIA GeForce RTX 3070".to_string(), 5888),
        ("NVIDIA GeForce RTX 2080 Ti".to_string(), 4352),
        ("NVIDIA GeForce RTX 2080 SUPER".to_string(), 3072),
        ("NVIDIA GeForce RTX 2080".to_string(), 2944),
        ("NVIDIA GeForce RTX 2070 SUPER".to_string(), 2560),
        ("NVIDIA GeForce GTX 1080 Ti".to_string(), 3584),
        ("NVIDIA GeForce GTX 1080".to_string(), 2560),
        ("NVIDIA GeForce GTX 2060".to_string(), 1920),
        ("NVIDIA GeForce GTX 1660 Ti".to_string(), 1536),
        ("NVIDIA GeForce GTX 1060".to_string(), 1280),
        ("NVIDIA GeForce GTX 1650 SUPER".to_string(), 1280),
        ("NVIDIA GeForce GTX 1650".to_string(), 896),
    ]
    .into_iter()
    .collect();

    if let Ok(var) = env::var("RUST_GPU_TOOLS_CUSTOM_GPU") {
        for card in var.split(',') {
            let split = card.rsplitn(2, ':').collect::<Vec<_>>();
            if split.len() != 2 {
                panic!("Invalid RUST_GPU_TOOLS_CUSTOM_GPU!");
            }
            let name = split[1].trim().to_string();
            let cores: usize = split[0]
                .trim()
                .parse()
                .expect("Invalid RUST_GPU_TOOLS_CUSTOM_GPU!");

            info!("Adding \"{}\" to GPU list with {} CUDA cores.", name, cores);
            core_counts.insert(name, cores);
        }
    }

    core_counts
}

#[cfg(test)]
mod tests {
    #[test]
    fn get_cuda_cores() {
        let core_counts = super::cuda_cores();
        let rtx_2080_ti_core_count = *core_counts.get("GeForce RTX 2080 Ti").unwrap();
        assert_eq!(rtx_2080_ti_core_count, 4352);
    }

    #[test]
    fn get_cuda_cores_missing() {
        let core_counts = super::cuda_cores();
        let unknown_core_count = core_counts.get("My unknown GPU").is_none();
        assert!(unknown_core_count);
    }

    #[test]
    fn get_cuda_cores_custom() {
        temp_env::with_var(
            "RUST_GPU_TOOLS_CUSTOM_GPU",
            Some("My custom GPU:12345,My other GPU:4444"),
            || {
                let core_counts = super::cuda_cores();
                let custom_core_count = *core_counts.get("My custom GPU").unwrap();
                assert_eq!(custom_core_count, 12345);
            },
        );
    }

    #[test]
    fn get_cuda_cores_custom_with_colons() {
        temp_env::with_var(
            "RUST_GPU_TOOLS_CUSTOM_GPU",
            Some("gfx906:sramecc-:xnack-:3840"),
            || {
                let core_counts = super::cuda_cores();
                let custom_core_count = *core_counts.get("gfx906:sramecc-:xnack-").unwrap();
                assert_eq!(custom_core_count, 3840);
            },
        );
    }
}
