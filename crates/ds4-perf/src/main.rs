#![forbid(unsafe_code)]

mod bench;
mod cli;
mod compare;
mod csv;
mod cupti;
mod doctor;
mod experiment;
mod fit;
mod inspect;
mod knobs;
mod ncu;
mod nsys;
mod optimize;
mod process;
mod report;
mod runner;
mod timeline;
mod workload;

use clap::Parser;

fn main() {
    if let Err(error) = process::signals() {
        eprintln!("ds4-perf: signal handler: {error}");
        std::process::exit(1);
    }
    let result = match cli::Cli::parse().command {
        cli::Command::Inspect(args) => inspect::run(
            &args.out,
            args.device,
            args.calibrate,
            args.gpu_helper.as_deref(),
            args.bench.as_deref(),
        ),
        cli::Command::Doctor { bench } => {
            print!(
                "{}",
                doctor::inspect(&mut runner::System, Some(&bench)).render()
            );
            Ok(())
        }
        cli::Command::Scout(args) => runner::scout(&args),
        cli::Command::Compare(args) => compare::run(&args),
        cli::Command::Optimize(args) => optimize::run(&args),
    };
    if let Err(error) = result {
        eprintln!("ds4-perf: {error}");
        std::process::exit(1);
    }
}
