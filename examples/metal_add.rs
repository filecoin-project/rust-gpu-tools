use rust_gpu_tools::{metal, Device};

fn main() {
    // Enable this example only when Metal feature is enabled
    #[cfg(feature = "metal")]
    {
        // Skip if not running on macOS
        if !cfg!(target_os = "macos") {
            println!("This example requires macOS. Skipping...");
            return;
        }

        // Find Metal devices
        let devices = Device::all();
        println!("Found {} GPU device(s)", devices.len());
        
        // Find the first Metal device
        let metal_devices: Vec<_> = devices
            .into_iter()
            .filter(|d| d.metal_device().is_some())
            .collect();
            
        if metal_devices.is_empty() {
            println!("No Metal devices found. Skipping...");
            return;
        }
        
        // Use the first Metal device
        let device = metal_devices[0];
        println!("Using device: {}", device.name());
        
        // Use Metal shader source directly using include_str!
        let src = include_str!("metal_add.metal");
        
        // Create Metal program
        let metal_device = device.metal_device().unwrap();
        let program = metal::Program::from_metal(metal_device, &src)
            .expect("Error creating Metal program");
        
        // Create input vectors
        const VECTOR_SIZE: usize = 1024;
        let a: Vec<f32> = (0..VECTOR_SIZE).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..VECTOR_SIZE).map(|i| (2*i) as f32).collect();
        let mut c = vec![0.0f32; VECTOR_SIZE];
        
        // Upload data to GPU
        let a_buffer = program.create_buffer_from_slice(&a)
            .expect("Error creating buffer A");
        let b_buffer = program.create_buffer_from_slice(&b)
            .expect("Error creating buffer B");
        let c_buffer = unsafe { 
            program.create_buffer::<f32>(VECTOR_SIZE)
                .expect("Error creating buffer C") 
        };
        
        // Create and run the kernel
        let threads_per_group = 256;
        let groups = ((VECTOR_SIZE + threads_per_group - 1) / threads_per_group) as usize;
        
        let kernel = program.create_kernel("add_vectors", groups, threads_per_group)
            .expect("Error creating kernel")
            .arg(&a_buffer)
            .arg(&b_buffer)
            .arg(&c_buffer);
            
        println!("Running kernel...");
        kernel.run().expect("Error running kernel");
        
        // Read back results
        program.read_into_buffer(&c_buffer, &mut c)
            .expect("Error reading result buffer");
            
        // Verify results
        println!("Verifying results...");
        let mut all_correct = true;
        for i in 0..VECTOR_SIZE {
            let expected = a[i] + b[i];
            if (c[i] - expected).abs() > 1e-5 {
                println!("Error at index {}: {} + {} = {} (expected {})",
                         i, a[i], b[i], c[i], expected);
                all_correct = false;
                break;
            }
        }
        
        if all_correct {
            println!("All results are correct!");
        }
    }
    
    #[cfg(not(feature = "metal"))]
    {
        println!("This example requires the Metal feature to be enabled.");
        println!("Compile with: cargo run --example metal/add --features=metal");
    }
}