//! Bench probe: load ASICamera2.dll and enumerate cameras exactly the way
//! the app's open_asi path does. Run from the repo root:
//!   cargo run --release -p skytracker-camera --example asi_probe [dll_path]

fn main() {
    let dll = std::env::args().nth(1).unwrap_or_else(|| "lib/ASICamera2.dll".into());
    println!("cwd: {}", std::env::current_dir().unwrap().display());
    println!("loading {dll} ...");
    let sdk = match skytracker_camera::asi::AsiSdk::load(&dll) {
        Ok(s) => {
            println!("DLL loaded OK");
            s
        }
        Err(e) => {
            println!("LOAD FAILED: {}", e.0);
            return;
        }
    };
    match sdk.num_cameras() {
        Ok(n) => {
            println!("connected cameras: {n}");
            for i in 0..n {
                match sdk.camera_info(i) {
                    Ok(info) => {
                        println!(
                            "  cam {i}: id={} {}x{}",
                            info.camera_id, info.max_width, info.max_height
                        );
                        match sdk.open(info.camera_id) {
                            Ok(()) => {
                                println!("    open+init OK");
                                let _ = sdk.close(info.camera_id);
                            }
                            Err(e) => println!("    open FAILED: {}", e.0),
                        }
                    }
                    Err(e) => println!("  cam {i}: info FAILED: {}", e.0),
                }
            }
        }
        Err(e) => println!("num_cameras FAILED: {}", e.0),
    }
}
