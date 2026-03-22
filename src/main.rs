mod ffi;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args[1] != "--model" {
        eprintln!("Usage: eabitnet --model <path.gguf> --prompt <text>");
        std::process::exit(1);
    }
    println!("eabitnet: model={}", args[2]);
    println!("Kernels linked successfully.");
}
