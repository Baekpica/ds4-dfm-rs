#![forbid(unsafe_code)]

mod bench;
mod cli;
mod csv;
mod doctor;
mod nsys;
mod report;
mod runner;
mod timeline;

fn main() {
    match cli::parse(std::env::args_os().skip(1)) {
        Ok(cli::Args::Help) => print!("{}", cli::USAGE),
        Ok(cli::Args::Doctor { bench }) => print!(
            "{}",
            doctor::inspect(&mut runner::System, Some(&bench)).render()
        ),
        Ok(cli::Args::Scout { out, command }) => {
            if let Err(err) = runner::scout(&out, &command) {
                eprintln!("ds4-perf: {err}");
                std::process::exit(1);
            }
        }
        Err(err) => {
            eprintln!("ds4-perf: {err}");
            std::process::exit(2);
        }
    }
}
