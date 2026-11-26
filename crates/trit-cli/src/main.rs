//! `trit`: generate stories and measure the engine from the command line.

use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

use trit_kernels::{Isa, Layout};
use trit_model::{Generator, Model, Sampler};

const USAGE: &str = "\
trit: a 1.58-bit TinyStories model

USAGE:
    trit generate [OPTIONS]       write a story
    trit info [--model PATH]      describe a model file
    trit bench [OPTIONS]          generation speed for every kernel and layout

OPTIONS:
    --model PATH        .trit file [default: models/tinystories-15m.trit]
    --prompt TEXT       how the story starts [default: \"Once upon a time\"]
    --temperature T     0 is greedy [default: 0.8]
    --top-p P           nucleus sampling [default: 0.9]
    --seed N            [default: random]
    --max-tokens N      stop after N tokens [default: 400]
    --layout i2|tq      in-memory weight layout [default: i2]
    --isa NAME          scalar, avx2, avx512-vnni [default: best available]
    --json PATH         (bench) also write the results as JSON
";

struct Args {
    cmd: String,
    model: String,
    prompt: String,
    temperature: f32,
    top_p: f32,
    seed: Option<u64>,
    max_tokens: usize,
    layout: Layout,
    isa: Option<Isa>,
    json: Option<String>,
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().ok_or("missing command")?;
    let mut a = Args {
        cmd,
        model: "models/tinystories-15m.trit".into(),
        prompt: "Once upon a time".into(),
        temperature: 0.8,
        top_p: 0.9,
        seed: None,
        max_tokens: 400,
        layout: Layout::I2,
        isa: None,
        json: None,
    };
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or(format!("{flag} needs a value"));
        let num = |s: String| {
            s.parse::<f64>()
                .map_err(|_| format!("{flag}: not a number: {s}"))
        };
        match flag.as_str() {
            "--model" => a.model = val()?,
            "--prompt" => a.prompt = val()?,
            "--temperature" => a.temperature = num(val()?)? as f32,
            "--top-p" => a.top_p = num(val()?)? as f32,
            "--seed" => a.seed = Some(val()?.parse().map_err(|_| "--seed: not an integer")?),
            "--max-tokens" => a.max_tokens = num(val()?)? as usize,
            "--layout" => {
                a.layout = match val()?.as_str() {
                    "i2" => Layout::I2,
                    "tq" => Layout::Tq,
                    other => return Err(format!("unknown layout {other}")),
                }
            }
            "--isa" => {
                let name = val()?;
                let isa = Isa::available()
                    .into_iter()
                    .find(|i| i.name() == name)
                    .ok_or(format!(
                        "kernel {name} is not available here (have {:?})",
                        Isa::available()
                    ))?;
                a.isa = Some(isa);
            }
            "--json" => a.json = Some(val()?),
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown option {other}")),
        }
    }
    Ok(a)
}

fn load(a: &Args, layout: Layout) -> Result<Model, String> {
    let bytes = std::fs::read(&a.model).map_err(|e| format!("{}: {e}", a.model))?;
    let mut m = Model::load(&bytes, layout).map_err(|e| format!("{}: {e}", a.model))?;
    if let Some(isa) = a.isa {
        m.isa = isa;
    }
    Ok(m)
}

fn generate(a: &Args) -> Result<(), String> {
    let model = load(a, a.layout)?;
    let seed = a.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(1, |d| d.as_nanos() as u64)
    });
    let mut g = Generator::new(&model, Sampler::new(a.temperature, a.top_p, seed));
    let t0 = Instant::now();
    let n_prompt = g.prompt(&a.prompt);
    let t1 = Instant::now();
    let mut out = std::io::stdout().lock();
    write!(out, "{}", a.prompt).ok();
    while g.generated < a.max_tokens {
        let Some(text) = g.next_text() else { break };
        write!(out, "{text}").ok();
        out.flush().ok();
    }
    writeln!(out).ok();
    let gen_s = t1.elapsed().as_secs_f64();
    eprintln!(
        "\n[{} prompt tokens in {:.1} ms, {} generated at {:.0} tok/s; seed {seed}, {} {}]",
        n_prompt + 1,
        (t1 - t0).as_secs_f64() * 1e3,
        g.generated,
        g.generated as f64 / gen_s,
        model.layers[0].wq.w.layout().name(),
        model.isa.name(),
    );
    Ok(())
}

fn info(a: &Args) -> Result<(), String> {
    let bytes = std::fs::read(&a.model).map_err(|e| format!("{}: {e}", a.model))?;
    let t = Instant::now();
    let model = Model::load(&bytes, Layout::I2).map_err(|e| format!("{}: {e}", a.model))?;
    let load_ms = t.elapsed().as_secs_f64() * 1e3;
    let c = model.config;
    let file = std::fs::metadata(&a.model).map(|m| m.len()).unwrap_or(0);
    let ternary = model.ternary_params();
    let embed = c.vocab_size * c.dim;
    println!("file            {} ({:.2} MB)", a.model, file as f64 / 1e6);
    println!(
        "shape           {} layers, dim {}, {} heads, ffn {}, vocab {}, context {}",
        c.n_layers, c.dim, c.n_heads, c.ffn_dim, c.vocab_size, c.max_seq_len
    );
    let widths: Vec<usize> = model.layers.iter().map(|l| l.w_up.w.rows()).collect();
    if widths.iter().any(|&w| w != c.ffn_dim) {
        let dead = c.n_layers * c.ffn_dim - widths.iter().sum::<usize>();
        println!("ffn per layer   {widths:?} ({dead} dead units pruned)");
    }
    println!(
        "ternary weights {ternary} ({:.2} MB at 1.6 bits)",
        ternary as f64 * 1.6 / 8e6
    );
    println!("int8 embedding  {embed} ({:.2} MB)", embed as f64 / 1e6);
    println!(
        "as f32          {:.1} MB",
        (ternary + embed) as f64 * 4.0 / 1e6
    );
    println!(
        "in memory (i2)  {:.2} MB",
        model.weight_bytes() as f64 / 1e6
    );
    println!("parse + repack  {load_ms:.1} ms");
    println!("kernels here    {:?}", Isa::available());
    Ok(())
}

fn bench(a: &Args) -> Result<(), String> {
    const STORIES: u64 = 5;
    let mut rows = Vec::new();
    for layout in Layout::ALL {
        let mut model = load(a, layout)?;
        for isa in Isa::available() {
            model.isa = isa;
            let mut tokens = 0;
            let mut secs = 0.0;
            for seed in 0..STORIES {
                let mut g = Generator::new(&model, Sampler::new(0.8, 0.9, seed));
                g.prompt(&a.prompt);
                let t = Instant::now();
                while g.generated < a.max_tokens.min(256) && g.next_token().is_some() {}
                secs += t.elapsed().as_secs_f64();
                tokens += g.generated;
            }
            let tps = tokens as f64 / secs;
            println!(
                "{:<3} {:<12} {tokens:>5} tokens  {tps:>8.0} tok/s",
                layout.name(),
                isa.name()
            );
            rows.push(format!(
                r#"{{"layout":"{}","kernel":"{}","tokens":{tokens},"seconds":{secs:.4},"tok_per_s":{tps:.1}}}"#,
                layout.name(),
                isa.name()
            ));
        }
    }
    if let Some(path) = &a.json {
        std::fs::write(path, format!("[\n{}\n]\n", rows.join(",\n"))).map_err(|e| e.to_string())?;
        println!("wrote {path}");
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("error: {e}\n");
            }
            eprint!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let r = match args.cmd.as_str() {
        "generate" => generate(&args),
        "info" => info(&args),
        "bench" => bench(&args),
        other => Err(format!("unknown command {other}\n\n{USAGE}")),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
