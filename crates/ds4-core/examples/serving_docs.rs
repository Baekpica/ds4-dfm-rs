#[path = "../tests/support/serving_docs.rs"]
mod serving_docs;

fn main() {
    print!("{}", serving_docs::render());
}
