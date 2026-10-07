//! Render a saved Chat Completions request without opening a model.
use ds4_core::chat_template::{RenderClock, Template};
use ds4_server::{chat_input, parse_chat_request, ParseEnv};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: glm53_prompt TEMPLATE REQUEST OUTPUT".into());
    }
    let source = std::fs::read_to_string(&args[0])?;
    let body = std::fs::read_to_string(&args[1])?;
    let template = Template::compile(&source, RenderClock::Fixed(0))?;
    let parsed = parse_chat_request(&ParseEnv::default(), &body)
        .map_err(|error| format!("request: {error:?}"))?;
    let prompt = chat_input::render(&template, ds4_core::Variant::Glm53Flash as i32, &parsed)
        .map_err(|error| format!("render: {error:?}"))?;
    std::fs::write(&args[2], prompt)?;
    Ok(())
}
