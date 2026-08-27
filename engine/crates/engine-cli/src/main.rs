use anyhow::Result;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 || args[1] != "--model" {
        eprintln!("Usage: engine-cli --model <dir> --prompt \"<text>\"");
        std::process::exit(1);
    }

    let _model_dir = &args[2];
    let _prompt = args.iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| "Hello".to_string());

    // TODO: milestone 4 — load model, run inference, stream tokens
    eprintln!("engine-cli scaffold — model loading not yet implemented");

    Ok(())
}
