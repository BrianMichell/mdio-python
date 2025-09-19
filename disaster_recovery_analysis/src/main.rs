use std::env;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;
fn decompress_blosc(compressed_data: &[u8]) -> Result<Vec<u8>, String> {
    unsafe {
        // Get decompressed size first
        let mut nbytes = 0usize;
        let mut cbytes = 0usize;
        let mut blocksize = 0usize;
        
        blosc_sys::blosc_cbuffer_sizes(
            compressed_data.as_ptr() as *const std::ffi::c_void,
            &mut nbytes as *mut usize,
            &mut cbytes as *mut usize,
            &mut blocksize as *mut usize,
        );
        
        if nbytes == 0 {
            return Err("Invalid compressed data".to_string());
        }
        
        // Allocate output buffer
        let mut decompressed = vec![0u8; nbytes];
        
        // Decompress
        let result = blosc_sys::blosc_decompress(
            compressed_data.as_ptr() as *const std::ffi::c_void,
            decompressed.as_mut_ptr() as *mut std::ffi::c_void,
            nbytes,
        );
        
        if result < 0 {
            return Err(format!("Blosc decompression failed with code: {}", result));
        }
        
        decompressed.truncate(result as usize);
        Ok(decompressed)
    }
}

fn print_hexdump(data: &[u8], offset: usize, chunk_name: &str) {
    println!("=== {} ===", chunk_name);
    for (i, chunk) in data.chunks(16).enumerate() {
        let addr = offset + i * 16;
        
        // Print address
        print!("{:08x}  ", addr);
        
        // Print hex bytes
        for (j, &byte) in chunk.iter().enumerate() {
            if j == 8 {
                print!(" "); // Extra space in the middle
            }
            print!("{:02x} ", byte);
        }
        
        // Pad if chunk is less than 16 bytes
        if chunk.len() < 16 {
            for j in chunk.len()..16 {
                if j == 8 {
                    print!(" ");
                }
                print!("   ");
            }
        }
        
        // Print ASCII representation
        print!(" |");
        for &byte in chunk {
            if byte >= 32 && byte <= 126 {
                print!("{}", byte as char);
            } else {
                print!(".");
            }
        }
        println!("|");
    }
    println!();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() != 2 {
        eprintln!("Usage: {} <zarr_array_path>", args[0]);
        eprintln!("Example: {} /path/to/zarr/array", args[0]);
        std::process::exit(1);
    }
    
    let zarr_path = Path::new(&args[1]);
    
    // Verify the path exists
    if !zarr_path.exists() {
        eprintln!("Error: Path '{}' does not exist", zarr_path.display());
        std::process::exit(1);
    }
    
    println!("Reading Zarr array from: {}", zarr_path.display());
    println!("========================================");
    
    // Read zarr.json metadata
    let zarr_json_path = zarr_path.join("zarr.json");
    if !zarr_json_path.exists() {
        eprintln!("Error: zarr.json not found in {}", zarr_path.display());
        std::process::exit(1);
    }
    
    let metadata_content = fs::read_to_string(&zarr_json_path)?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata_content)?;
    
    // Extract information from metadata
    let shape = metadata["shape"].as_array().unwrap();
    let chunk_shape = metadata["chunk_grid"]["configuration"]["chunk_shape"].as_array().unwrap();
    
    println!("Array shape: {:?}", shape);
    println!("Chunk shape: {:?}", chunk_shape);
    println!("Data type: {}", metadata["data_type"]["name"]);
    if let Some(config) = metadata["data_type"]["configuration"].as_object() {
        if let Some(length_bytes) = config.get("length_bytes") {
            println!("Length bytes: {}", length_bytes);
        }
    }
    println!();
    
    // Calculate expected chunks based on the metadata we know:
    // Shape: [345, 188], Chunk shape: [128, 128]
    // This means we have ceil(345/128) = 3 chunks in dimension 0
    // and ceil(188/128) = 2 chunks in dimension 1
    // So we expect chunks: c/0/0, c/0/1, c/1/0, c/1/1, c/2/0, c/2/1
    
    let mut chunk_files = Vec::new();
    
    // Find all chunk files by walking the directory
    for entry in WalkDir::new(zarr_path) {
        let entry = entry?;
        let path = entry.path();
        
        // Look for chunk files (they start with 'c/' in Zarr v3)
        if path.is_file() {
            let relative_path = path.strip_prefix(zarr_path)?;
            let path_str = relative_path.to_string_lossy();
            
            if path_str.starts_with("c/") {
                chunk_files.push((path.to_path_buf(), path_str.to_string()));
            }
        }
    }
    
    // Sort chunk files for consistent ordering
    chunk_files.sort_by(|a, b| a.1.cmp(&b.1));
    
    println!("Found {} chunk files:", chunk_files.len());
    for (_, chunk_name) in &chunk_files {
        println!("  {}", chunk_name);
    }
    println!();
    
    let mut total_offset = 0;
    
    // Read, decompress, and hexdump each chunk file
    for (chunk_path, chunk_name) in chunk_files {
        match fs::read(&chunk_path) {
            Ok(compressed_data) => {
                if compressed_data.is_empty() {
                    println!("=== {} ===", chunk_name);
                    println!("(empty chunk)");
                    println!();
                } else {
                    println!("Compressed size: {} bytes", compressed_data.len());
                    
                    // Decompress the Blosc-compressed data using blosc-sys directly
                    match decompress_blosc(&compressed_data) {
                        Ok(decompressed_data) => {
                            println!("Decompressed size: {} bytes", decompressed_data.len());
                            print_hexdump(&decompressed_data, total_offset, &chunk_name);
                            total_offset += decompressed_data.len();
                        }
                        Err(e) => {
                            eprintln!("Error decompressing chunk {}: {}", chunk_name, e);
                            println!("Showing raw compressed data instead:");
                            print_hexdump(&compressed_data, total_offset, &format!("{} (compressed)", chunk_name));
                            total_offset += compressed_data.len();
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Error reading chunk {}: {}", chunk_name, e);
            }
        }
    }
    
    println!("Total decompressed bytes processed: {}", total_offset);
    println!();
    println!("Note: This shows the decompressed array data as it would appear in memory.");
    println!("Each element is 240 bytes (raw_bytes with length_bytes: 240).");
    
    Ok(())
}
