fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let result = match args.as_slice() {
        [command, ordinal] if command == "inspect" => ordinal
            .parse::<usize>()
            .map_err(|e| e.to_string())
            .and_then(ds4_perf_gpu::device::inspect)
            .and_then(|gpu| serde_json::to_string(&gpu).map_err(|e| e.to_string())),
        [command, ordinal] if command == "calibrate" => ordinal
            .parse::<usize>()
            .map_err(|e| e.to_string())
            .and_then(ds4_perf_gpu::calibration::run)
            .and_then(|calibration| serde_json::to_string(&calibration).map_err(|e| e.to_string())),
        _ => Err("usage: ds4-perf-gpu inspect|calibrate DEVICE".into()),
    };
    match result {
        Ok(text) => println!("{text}"),
        Err(error) => {
            eprintln!("ds4-perf-gpu: {error}");
            std::process::exit(1);
        }
    }
}
