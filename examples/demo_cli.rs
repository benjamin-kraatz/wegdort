use std::env;
use std::path::Path;
use wegdort::{Metric, Store, StoreBuilder, VectorId};

const ITEMS: &[(u64, &str, [f32; 3])] = &[
    (1, "coffee", [0.95, 0.10, 0.05]),
    (2, "espresso", [0.90, 0.18, 0.08]),
    (3, "tea", [0.72, 0.42, 0.12]),
    (4, "laptop", [0.05, 0.92, 0.20]),
    (5, "keyboard", [0.08, 0.82, 0.36]),
    (6, "notebook", [0.18, 0.62, 0.65]),
    (7, "bicycle", [0.12, 0.20, 0.94]),
    (8, "running shoes", [0.20, 0.16, 0.88]),
];

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        eprintln!();
        print_usage();
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "help".to_string());

    match command.as_str() {
        "help" | "-h" | "--help" => print_usage(),
        "list" => list_items(),
        "search" => {
            let metric = parse_metric(args.next().as_deref())?;
            let k = parse_k(args.next().as_deref())?;
            let query = parse_query(args.collect::<Vec<_>>())?;
            let store = seed_store(metric)?;
            print_hits(&store, &query, k)?;
        }
        "compare" => {
            let k = parse_k(args.next().as_deref())?;
            let query = parse_query(args.collect::<Vec<_>>())?;
            for metric in [Metric::Cosine, Metric::Dot, Metric::SquaredL2] {
                println!("\n{metric:?}");
                let store = seed_store(metric)?;
                print_hits(&store, &query, k)?;
            }
        }
        "snapshot" => {
            let metric = parse_metric(args.next().as_deref())?;
            let k = parse_k(args.next().as_deref())?;
            let query = parse_query(args.collect::<Vec<_>>())?;
            let mut store = seed_store(metric)?;
            let snapshot = store.snapshot();

            store.upsert(VectorId::new(99), [0.0, 0.0, 1.0])?;

            println!("snapshot results before later store mutation:");
            for hit in snapshot.search(query, k)? {
                println!("  {}", format_hit(hit.id, hit.score));
            }

            println!("\ncurrent store results after mutation:");
            print_hits(&store, &query, k)?;
        }
        "save" => {
            let path = args.next().ok_or("missing snapshot path")?;
            let metric = parse_metric(args.next().as_deref())?;
            let store = seed_store(metric)?;
            store.save(Path::new(&path))?;
            println!(
                "saved {} vectors with {metric:?} metric to {path}",
                store.len()
            );
        }
        "load" => {
            let path = args.next().ok_or("missing snapshot path")?;
            let k = parse_k(args.next().as_deref())?;
            let query = parse_query(args.collect::<Vec<_>>())?;
            let store = Store::load(Path::new(&path))?;
            println!(
                "loaded {} vectors, dimensions={}, metric={:?}",
                store.len(),
                store.dimensions(),
                store.metric()
            );
            print_hits(&store, &query, k)?;
        }
        #[cfg(feature = "parallel")]
        "parallel" => {
            let metric = parse_metric(args.next().as_deref())?;
            let k = parse_k(args.next().as_deref())?;
            let query = parse_query(args.collect::<Vec<_>>())?;
            let store = seed_store(metric)?;
            for hit in store.search_parallel(query, k)? {
                println!("{}", format_hit(hit.id, hit.score));
            }
        }
        _ => return Err(format!("unknown command '{command}'").into()),
    }

    Ok(())
}

fn seed_store(metric: Metric) -> Result<Store, wegdort::Error> {
    let mut store = StoreBuilder::new(3)
        .metric(metric)
        .capacity(ITEMS.len())
        .build()?;

    store.reserve(4);
    for (id, _label, vector) in ITEMS {
        store.insert(VectorId::new(*id), vector)?;
    }

    Ok(store)
}

fn list_items() {
    println!("demo vectors:");
    for (id, label, vector) in ITEMS {
        println!("  {id:>2}  {label:<14}  {:?}", vector);
    }
}

fn print_hits(store: &Store, query: &[f32; 3], k: usize) -> Result<(), wegdort::Error> {
    for hit in store.search(query, k)? {
        println!("{}", format_hit(hit.id, hit.score));
    }
    Ok(())
}

fn format_hit(id: VectorId, score: f32) -> String {
    let label = label_for(id).unwrap_or("<external metadata missing>");
    format!("id={:<2} label={:<28} score={:.6}", id.get(), label, score)
}

fn label_for(id: VectorId) -> Option<&'static str> {
    ITEMS
        .iter()
        .find(|(item_id, _, _)| *item_id == id.get())
        .map(|(_, label, _)| *label)
}

fn parse_metric(value: Option<&str>) -> Result<Metric, Box<dyn std::error::Error>> {
    match value {
        Some("cosine") => Ok(Metric::Cosine),
        Some("dot") => Ok(Metric::Dot),
        Some("l2") | Some("squared-l2") => Ok(Metric::SquaredL2),
        Some(value) => Err(format!("unknown metric '{value}'").into()),
        None => Err("missing metric: use cosine, dot, or l2".into()),
    }
}

fn parse_k(value: Option<&str>) -> Result<usize, Box<dyn std::error::Error>> {
    let value = value.ok_or("missing k")?;
    Ok(value.parse()?)
}

fn parse_query(values: Vec<String>) -> Result<[f32; 3], Box<dyn std::error::Error>> {
    if values.len() != 3 {
        return Err("query must contain exactly three f32 values".into());
    }

    Ok([values[0].parse()?, values[1].parse()?, values[2].parse()?])
}

fn print_usage() {
    println!(
        "\
wegdort demo CLI

Usage:
  cargo run --example demo_cli -- list
  cargo run --example demo_cli -- search <cosine|dot|l2> <k> <x> <y> <z>
  cargo run --example demo_cli -- compare <k> <x> <y> <z>
  cargo run --example demo_cli -- snapshot <cosine|dot|l2> <k> <x> <y> <z>
  cargo run --example demo_cli -- save <path> <cosine|dot|l2>
  cargo run --example demo_cli -- load <path> <k> <x> <y> <z>

With the parallel feature:
  cargo run --features parallel --example demo_cli -- parallel <cosine|dot|l2> <k> <x> <y> <z>

Examples:
  cargo run --example demo_cli -- search cosine 3 1.0 0.1 0.0
  cargo run --example demo_cli -- compare 3 0.1 0.2 1.0
  cargo run --example demo_cli -- save /tmp/wegdort-demo.wgd cosine
  cargo run --example demo_cli -- load /tmp/wegdort-demo.wgd 3 1.0 0.1 0.0
"
    );
}
