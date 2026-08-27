use anyhow::{Result, Context};
use candle_core::Device;
use engine_formats::{ModelConfig, SafetensorsLoader};
use engine_formats::tokenizer::Tokenizer;
use engine_runtime::{Model, Sampler, Weights};
use std::io::Write;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let model_dir = get_arg(&args, "--model")
        .context("usage: engine-cli --model <dir> [--prompt <text>] [--max-tokens <n>] [--temperature <t>]")?;
    let prompt = get_arg(&args, "--prompt").unwrap_or_else(|_| "Hello".to_string());
    let max_tokens: usize = get_arg(&args, "--max-tokens")
        .unwrap_or_else(|_| "128".to_string())
        .parse()?;
    let temperature: f32 = get_arg(&args, "--temperature")
        .unwrap_or_else(|_| "0.0".to_string())
        .parse()?;

    let model_path = Path::new(&model_dir);

    eprintln!("loading config...");
    let config = ModelConfig::from_json(&model_path.join("config.json"))?;
    eprintln!("  {} layers, {} heads, dim {}", config.num_hidden_layers, config.num_attention_heads, config.hidden_size);

    eprintln!("loading tokenizer...");
    let tokenizer = Tokenizer::from_file(&model_path.join("tokenizer.json"))?;

    eprintln!("loading weights...");
    let loader = SafetensorsLoader::new(model_path)?;
    let raw_tensors = loader.load_tensors()?;
    eprintln!("  {} tensors loaded", raw_tensors.len());

    let device = Device::Cpu;
    let weights = Weights::new(raw_tensors, device);

    eprintln!("building model...");
    let mut model = Model::load(config, weights)?;

    let sampler = if temperature == 0.0 {
        Sampler::Greedy
    } else {
        Sampler::Temperature { temp: temperature, top_p: 0.9 }
    };

    eprintln!("generating...\n");

    // Encode prompt
    let mut token_ids = tokenizer.encode(&prompt)?;
    let prompt_len = token_ids.len();

    // Prefill: process entire prompt
    let logits = model.forward(&token_ids, 0)?;
    let logits_data: Vec<f32> = logits.flatten_all()?.to_vec1()?;
    let next_token = sampler.sample(&logits_data);
    token_ids.push(next_token);

    // Print prompt
    print!("{}", prompt);
    // Print first generated token
    print!("{}", tokenizer.decode(&[next_token])?);
    std::io::stdout().flush()?;

    // Decode loop: one token at a time
    for i in 0..max_tokens - 1 {
        let pos = prompt_len + i;
        let logits = model.forward(&[*token_ids.last().unwrap()], pos + 1)?;
        let logits_data: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let next_token = sampler.sample(&logits_data);

        if next_token == 2 || next_token == 0 {
            break; // EOS
        }

        token_ids.push(next_token);
        print!("{}", tokenizer.decode(&[next_token])?);
        std::io::stdout().flush()?;
    }

    println!();
    eprintln!("\n[generated {} tokens]", token_ids.len() - prompt_len);

    Ok(())
}

fn get_arg(args: &[String], flag: &str) -> Result<String> {
    let pos = args.iter().position(|a| a == flag)
        .ok_or_else(|| anyhow::anyhow!("missing flag: {flag}"))?;
    args.get(pos + 1)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing value for {flag}"))
}
