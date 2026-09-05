//! C↔Rust tokenizer encode / render / decode / stop / specials.
//! Synthetic metadata-only GGUF v3. Does not open a production model.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use ds4_core::{dump_cmd, ChatThinkMode, ModelFamily, TokError, TokenBuffer, Vocab};

fn vocab_oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_VOCAB_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/vocab_c_oracle")
}

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_TOKENIZER_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/tokenizer_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/tokenizer_c_oracle (missing {})",
        p.display()
    );
    p
}

fn gpt2_byte_to_codepoint(b: u8) -> u32 {
    if (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174 {
        return u32::from(b);
    }
    let mut n = 0u32;
    for x in 0u32..256 {
        let xb = x as u8;
        if (33..=126).contains(&xb) || (161..=172).contains(&xb) || xb >= 174 {
            continue;
        }
        if xb == b {
            return 256 + n;
        }
        n += 1;
    }
    u32::from(b)
}

fn utf8_put(out: &mut Vec<u8>, cp: u32) {
    if cp <= 0x7f {
        out.push(cp as u8);
    } else if cp <= 0x7ff {
        out.push((0xc0 | (cp >> 6)) as u8);
        out.push((0x80 | (cp & 0x3f)) as u8);
    } else if cp <= 0xffff {
        out.push((0xe0 | (cp >> 12)) as u8);
        out.push((0x80 | ((cp >> 6) & 0x3f)) as u8);
        out.push((0x80 | (cp & 0x3f)) as u8);
    } else {
        out.push((0xf0 | (cp >> 18)) as u8);
        out.push((0x80 | ((cp >> 12) & 0x3f)) as u8);
        out.push((0x80 | ((cp >> 6) & 0x3f)) as u8);
        out.push((0x80 | (cp & 0x3f)) as u8);
    }
}

fn gpt2_byte_token(b: u8) -> Vec<u8> {
    let mut out = Vec::new();
    utf8_put(&mut out, gpt2_byte_to_codepoint(b));
    out
}

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_bytes(buf: &mut Vec<u8>, s: &[u8]) {
    put_u64(buf, s.len() as u64);
    buf.extend_from_slice(s);
}

fn write_tok_gguf(
    path: &Path,
    arch: &str,
    tokens: &[Vec<u8>],
    merges: &[Vec<u8>],
    types: &[u32],
    type_tag: u32,
    special_ids: Option<(u32, u32)>,
) {
    assert_eq!(tokens.len(), types.len());
    let mut buf = Vec::new();
    put_u32(&mut buf, 0x4655_4747);
    put_u32(&mut buf, 3);
    put_u64(&mut buf, 0);
    put_u64(&mut buf, 4 + u64::from(special_ids.is_some()) * 2);
    put_bytes(&mut buf, b"general.architecture");
    put_u32(&mut buf, 8);
    put_bytes(&mut buf, arch.as_bytes());
    put_bytes(&mut buf, b"tokenizer.ggml.tokens");
    put_u32(&mut buf, 9);
    put_u32(&mut buf, 8);
    put_u64(&mut buf, tokens.len() as u64);
    for t in tokens {
        put_bytes(&mut buf, t);
    }
    put_bytes(&mut buf, b"tokenizer.ggml.merges");
    put_u32(&mut buf, 9);
    put_u32(&mut buf, 8);
    put_u64(&mut buf, merges.len() as u64);
    for m in merges {
        put_bytes(&mut buf, m);
    }
    put_bytes(&mut buf, b"tokenizer.ggml.token_type");
    put_u32(&mut buf, 9);
    put_u32(&mut buf, type_tag);
    put_u64(&mut buf, types.len() as u64);
    for t in types {
        buf.extend_from_slice(&t.to_le_bytes());
    }
    if let Some((bos, eos)) = special_ids {
        for (key, id) in [
            (b"tokenizer.ggml.bos_token_id".as_slice(), bos),
            (b"tokenizer.ggml.eos_token_id".as_slice(), eos),
        ] {
            put_bytes(&mut buf, key);
            put_u32(&mut buf, 4);
            put_u32(&mut buf, id);
        }
    }
    fs::write(path, buf).unwrap();
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("ds4-tokenizer-parity");
    fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn hex_text(s: &str) -> String {
    s.as_bytes().iter().map(|b| format!("{b:02x}")).collect()
}

fn c_cmd(family: &str, path: &Path, cmd: &str, arg: &str) -> String {
    let mut args = vec![
        family.to_string(),
        path.display().to_string(),
        cmd.to_string(),
    ];
    if cmd != "specials" {
        args.push(arg.to_string());
    }
    let out = Command::new(require_oracle())
        .args(&args)
        .output()
        .expect("run tokenizer_c_oracle");
    assert!(
        out.status.success(),
        "oracle {family} {cmd} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("oracle utf8")
}

fn assert_cmd(family: ModelFamily, path: &Path, vocab: &Vocab, cmd: &str, arg: &str) {
    let fam = family.oracle_name();
    let c = c_cmd(fam, path, cmd, arg);
    let rust = dump_cmd(vocab, cmd, arg);
    assert_eq!(rust, c, "mismatch {fam} {cmd} {arg}");
}

fn rust_chat_cmd(vocab: &Vocab, mode: ChatThinkMode) -> Result<String, TokError> {
    let mut tokens = TokenBuffer::new();
    vocab.chat_begin(&mut tokens)?;
    vocab.chat_append_effort_prefix(&mut tokens, mode);
    vocab.chat_append_message(&mut tokens, "system", b"Policy <think>system</think>.")?;
    vocab.chat_append_message(&mut tokens, "developer", b"Developer policy.")?;
    vocab.chat_append_message(&mut tokens, "user", b"hello")?;
    let assistant = if vocab.family == ModelFamily::SolarOpen2 {
        b"<|think:start|>trace<|think:end|>answer".as_slice()
    } else {
        b"<think>trace</think>answer".as_slice()
    };
    vocab.chat_append_message(&mut tokens, "assistant", assistant)?;
    if vocab.family == ModelFamily::SolarOpen2 {
        tokens.push(vocab.im_end_id);
    }
    vocab.chat_append_message(
        &mut tokens,
        "tool",
        b"A </tool_response> B </dots_function_response> C <|tool_response:end|> D",
    )?;
    vocab.chat_append_message(
        &mut tokens,
        "function",
        b"raw:\xff A </tool_response> B </dots_function_response> C <|tool_response:end|> D",
    )?;
    if vocab.family == ModelFamily::SolarOpen2 {
        tokens.push(vocab.im_end_id);
    }
    vocab.chat_append_assistant_prefix(&mut tokens, mode)?;
    let ids = tokens
        .as_slice()
        .iter()
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    Ok(format!("TOKENS {ids}\n"))
}

fn assert_chat_cmd(family: ModelFamily, path: &Path, vocab: &Vocab, mode: ChatThinkMode) {
    let c = c_cmd(
        family.oracle_name(),
        path,
        "chat",
        &(mode as i32).to_string(),
    );
    let rust = rust_chat_cmd(vocab, mode).expect("rust chat");
    assert_eq!(
        rust,
        c,
        "mismatch {} chat mode={mode:?}",
        family.oracle_name()
    );
}

struct Builder {
    tokens: Vec<Vec<u8>>,
    types: Vec<u32>,
    merges: Vec<Vec<u8>>,
}

impl Builder {
    fn with_bytes() -> Self {
        let mut tokens = Vec::with_capacity(320);
        let mut types = Vec::with_capacity(320);
        for b in 0..=255u8 {
            tokens.push(gpt2_byte_token(b));
            types.push(1);
        }
        Self {
            tokens,
            types,
            merges: vec![b"h e".to_vec()],
        }
    }

    fn motif_added() -> Self {
        let mut tokens = Vec::with_capacity(480);
        let mut types = Vec::with_capacity(480);
        for i in 0..160 {
            tokens.push(format!("<a{i:03}>").into_bytes());
            types.push(3);
        }
        for (i, s) in [
            "<|beginoftext|>",
            "<|endoftext|>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|startofturn|>",
            "<|endofturn|>",
            "<|tool|>",
            "<|reference|>",
            "<|plan|>",
            "<|endofplan|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<latent>",
            "<latent_pad>",
            "</latent>",
        ]
        .into_iter()
        .enumerate()
        {
            tokens[i] = s.as_bytes().to_vec();
        }
        tokens[53] = b"<ls53>".to_vec();
        tokens[54] = b"<ls54>".to_vec();
        tokens[85] = b"<ls85>".to_vec();
        for b in 0..=255u8 {
            tokens.push(gpt2_byte_token(b));
            types.push(1);
        }
        Self {
            tokens,
            types,
            merges: vec![b"h e".to_vec()],
        }
    }

    fn push(&mut self, s: &[u8], typ: u32) -> i32 {
        let id = self.tokens.len() as i32;
        self.tokens.push(s.to_vec());
        self.types.push(typ);
        id
    }

    fn push_str(&mut self, s: &str, typ: u32) -> i32 {
        self.push(s.as_bytes(), typ)
    }

    fn write(&self, path: &Path, arch: &str) {
        write_tok_gguf(path, arch, &self.tokens, &self.merges, &self.types, 4, None);
    }

    fn write_specials(&self, path: &Path, arch: &str, bos: i32, eos: i32) {
        write_tok_gguf(
            path,
            arch,
            &self.tokens,
            &self.merges,
            &self.types,
            4,
            Some((bos as u32, eos as u32)),
        );
    }

    fn write_types(&self, path: &Path, arch: &str, type_tag: u32) {
        write_tok_gguf(
            path,
            arch,
            &self.tokens,
            &self.merges,
            &self.types,
            type_tag,
            None,
        );
    }
}

impl Builder {
    fn with_he(mut self) -> Self {
        if !self.tokens.iter().any(|t| t == b"he") {
            self.push_str("he", 1);
        }
        self
    }
}

fn write_family(family: ModelFamily) -> PathBuf {
    let path = tmp(&format!("{}.gguf", family.oracle_name()));
    match family {
        ModelFamily::Glm53 => {
            let mut b = Builder::with_bytes().with_he();
            let bos = b.push_str("[gMASK]", 3);
            let eos = b.push_str("<|endoftext|>", 3);
            b.push_str("<|system|>", 3);
            b.push_str("<|user|>", 3);
            b.push_str("<|assistant|>", 3);
            b.push_str("<|observation|>", 3);
            b.push_str("<sop>", 3);
            b.push_str("<think>", 3);
            b.push_str("</think>", 3);
            b.push_str("<tool_call>", 3);
            b.push_str("</tool_call>", 3);
            b.push_str("<tool_response>", 3);
            b.push_str("</tool_response>", 3);
            b.push_str("<arg_key>", 3);
            b.push_str("</arg_key>", 3);
            b.push_str("<arg_value>", 3);
            b.push_str("</arg_value>", 3);
            b.write_specials(&path, "glm5-next", bos, eos);
        }
        ModelFamily::DeepSeek4 => {
            let mut b = Builder::with_bytes().with_he();
            b.push_str("<｜begin▁of▁sentence｜>", 3);
            b.push_str("<｜end▁of▁sentence｜>", 3);
            b.push_str("<｜User｜>", 3);
            b.push_str("<｜Assistant｜>", 3);
            b.push_str("<think>", 3);
            b.push_str("</think>", 3);
            b.push_str("｜DSML｜", 3);
            b.write(&path, "deepseek4");
        }
        ModelFamily::Motif3 => {
            let b = Builder::motif_added().with_he();
            b.write(&path, "motif3");
        }
        ModelFamily::SolarOpen2 => {
            let mut b = Builder::with_bytes().with_he();
            b.push_str("<|startoftext|>", 3);
            b.push_str("<|endoftext|>", 3);
            b.push_str("<|im:start|>", 3);
            b.push_str("<|im:content|>", 3);
            b.push_str("<|im:end|>", 3);
            b.push_str("<|think:start|>", 3);
            b.push_str("<|think:end|>", 3);
            b.push_str("<|tool:start|>", 3);
            b.push_str("<|tool:end|>", 3);
            b.push_str("<|tool_call:start|>", 3);
            b.push_str("<|tool_call:end|>", 3);
            b.push_str("<|tool_response:start|>", 3);
            b.push_str("<|tool_response:end|>", 3);
            b.push_str("<ud>", 4);
            b.write(&path, "solar-open2");
        }
        ModelFamily::ExaoneMoe => {
            let mut b = Builder::with_bytes().with_he();
            b.push_str("[BOS]", 3);
            b.push_str("<|endofturn|>", 3);
            b.push_str("<|system|>", 3);
            b.push_str("<|user|>", 3);
            b.push_str("<|assistant|>", 3);
            b.push_str("<|tool|>", 3);
            b.push_str("<think>", 3);
            b.push_str("</think>", 3);
            b.push_str("<tool_call>", 3);
            b.push_str("</tool_call>", 3);
            b.push_str("<tool_result>", 3);
            b.push_str("</tool_result>", 3);
            b.push_str("    ", 4);
            b.push_str("<ud>", 4);
            b.write(&path, "exaone-moe");
        }
        ModelFamily::Dots3Note => {
            let mut b = Builder::with_bytes().with_he();
            b.push_str("<|endoftext|>", 3);
            b.push_str("<|endofassistant|>", 3);
            b.push_str("<|system|>", 3);
            b.push_str("<|user|>", 3);
            b.push_str("<|assistant|>", 3);
            b.push_str("<|endofsystem|>", 3);
            b.push_str("<|endofuser|>", 3);
            b.push_str("<think>", 3);
            b.push_str("</think>", 3);
            b.push_str("<dots_function_call>", 3);
            b.push_str("</dots_function_call>", 3);
            b.push_str("<dots_function_response>", 3);
            b.push_str("</dots_function_response>", 3);
            b.push_str("<no_think>", 4);
            b.write(&path, "dots3-note");
        }
        ModelFamily::Qwen4Exp => {
            let mut b = Builder::with_bytes().with_he();
            b.push_str("<|endoftext|>", 3);
            b.push_str("<|im_start|>", 3);
            b.push_str("<|im_end|>", 3);
            b.push_str("<think>", 3);
            b.push_str("</think>", 3);
            b.push_str("<tool_call>", 3);
            b.push_str("</tool_call>", 3);
            b.push_str("<tool_response>", 3);
            b.push_str("</tool_response>", 3);
            b.write(&path, "qwen4exp");
        }
    }
    path
}

fn load(family: ModelFamily, path: &Path) -> Vocab {
    Vocab::load_path(path, family).expect("rust vocab_load")
}

#[test]
fn k2_horizon_tokenizer_matches_reference_vectors() {
    let Ok(path) = std::env::var("DS4_K2_MODEL") else {
        return;
    };
    let vocab =
        Vocab::load_path(Path::new(&path), ModelFamily::ExaoneMoe).expect("load K2 tokenizer");
    let cases: &[(&str, &[i32])] = &[
        ("hello", &[33785]),
        ("12345", &[4018, 927]),
        ("안녕하세요", &[76943, 47589, 245, 187709]),
        ("a\u{200c}b", &[66, 37387, 67]),
        ("a\u{200d}b", &[66, 16727, 67]),
        /* GGUF/llama.cpp intentionally does not apply tokenizer.json's NFC
         * normalizer here; pin the deployed GGUF behavior, not HF Python. */
        ("Cafe\u{301} noir", &[164494, 53986, 70603]),
    ];
    for (text, expected) in cases {
        assert_eq!(vocab.encode_text(text), *expected, "{text:?}");
    }
    let rendered = concat!(
        "<|ifm|begin_of_text|><|ifm|im_start|>user\n",
        "안녕<|ifm|im_end|><|ifm|im_start|>assistant\n",
        "<ifm|think>\n"
    );
    assert_eq!(
        vocab.encode_rendered_chat(rendered),
        vec![0, 250018, 2672, 200, 76943, 47589, 245, 250019, 250018, 142036, 200, 250029, 200,]
    );
}

fn family_cases(family: ModelFamily) {
    let path = write_family(family);
    let vocab = load(family, &path);
    assert_cmd(family, &path, &vocab, "specials", "");

    let mut encodes: Vec<&str> = vec![
        "",
        "hello",
        "12345",
        ">;\n",
        "  hello",
        "hello world",
        "it's",
    ];
    let mut renders: Vec<String> = Vec::new();
    match family {
        ModelFamily::Glm53 => {
            encodes.extend(["12345", "HelloWorld", "안녕하세요"]);
            renders.push("[gMASK]<sop><|user|>hello<|assistant|>".into());
            renders.push("<think>plan</think>".into());
            renders.push(
                "<tool_call>run<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>".into(),
            );
            renders.push("<|begin_of_image|><|image|><|end_of_image|>".into());
        }
        ModelFamily::DeepSeek4 => {
            encodes.push("你好");
            renders.push("<｜User｜>hello".into());
            renders.push("<think>hi</think>".into());
            renders.push("｜DSML｜x".into());
        }
        ModelFamily::Motif3 => {
            encodes.extend([
                "<|user|>hello",
                "hello  <ls53>x",
                "hello  <ls85>x",
                "HelloWorld",
            ]);
            renders.push("<|user|>hello<|endofturn|>".into());
            renders.push("<think>plan</think>".into());
        }
        ModelFamily::SolarOpen2 => {
            encodes.push("hello<ud>");
            renders.push("<|im:start|>user<|im:content|>hi<|im:end|>".into());
            renders.push("<|think:start|>x<|think:end|>".into());
        }
        ModelFamily::ExaoneMoe => {
            encodes.extend(["    hello", "hello<ud>", "hello\nworld"]);
            renders.push("<|user|>hello<|endofturn|>".into());
            renders.push("<think>x</think>".into());
        }
        ModelFamily::Dots3Note => {
            encodes.extend([" <no_think>", "hi<no_think>there"]);
            renders.push("<|user|>hi<|endofassistant|>".into());
            renders.push("<|endoftext|>".into());
        }
        ModelFamily::Qwen4Exp => {
            encodes.push("hello <think>x</think>");
            renders.push("<|im_start|>user\nhi<|im_end|>\n".into());
            renders.push("<think>x</think>".into());
        }
    }
    for t in encodes {
        assert_cmd(family, &path, &vocab, "encode", &hex_text(t));
    }
    for t in &renders {
        assert_cmd(family, &path, &vocab, "render", &hex_text(t));
    }

    let decode_ids = [
        i32::from(b'h'),
        i32::from(b' '),
        vocab.bos_id,
        vocab.eos_id,
        vocab.engine_eos(),
        -1,
        99999,
    ];
    for id in decode_ids {
        assert_cmd(family, &path, &vocab, "decode", &id.to_string());
    }
    if family == ModelFamily::ExaoneMoe {
        /* four-space USER_DEFINED is not a GPT-2 byte token. */
        let id = vocab.encode_text("    ")[0];
        assert_cmd(family, &path, &vocab, "decode", &id.to_string());
    }

    let mut stops = vec![vocab.eos_id, vocab.engine_eos(), 7, -1];
    match family {
        ModelFamily::SolarOpen2 | ModelFamily::Qwen4Exp => stops.push(vocab.eot_id),
        ModelFamily::Motif3 => {
            stops.push(vocab.user_id);
            stops.push(vocab.end_of_turn_id);
            stops.push(vocab.bos_id);
        }
        ModelFamily::Dots3Note => {
            stops.push(vocab.dots3_endoftext_id);
            stops.push(vocab.user_id);
        }
        _ => {}
    }
    for id in stops {
        assert_cmd(family, &path, &vocab, "stop", &id.to_string());
    }
    for mode in [
        ChatThinkMode::None,
        ChatThinkMode::Low,
        ChatThinkMode::High,
        ChatThinkMode::Max,
    ] {
        assert_chat_cmd(family, &path, &vocab, mode);
    }

    let mut tokens = TokenBuffer::from_tokens(vec![7]);
    match family {
        ModelFamily::SolarOpen2 => {
            let mut missing = vocab.clone();
            missing.tool_response_start_id = -1;
            let err = missing
                .chat_append_message(&mut tokens, "tool", b"result")
                .unwrap_err();
            assert!(
                matches!(err, TokError::MissingToken(ref token) if token == "<|tool_response:start|>")
            );
        }
        ModelFamily::ExaoneMoe => {
            let mut missing = vocab.clone();
            missing.bos_id = -1;
            let err = missing.chat_begin(&mut tokens).unwrap_err();
            assert!(matches!(err, TokError::MissingToken(ref token) if token == "[BOS]"));

            missing.bos_id = vocab.bos_id;
            missing.assistant_id = -1;
            let err = missing
                .chat_append_assistant_prefix(&mut tokens, ChatThinkMode::Low)
                .unwrap_err();
            assert!(matches!(err, TokError::MissingToken(ref token) if token == "<|assistant|>"));

            missing.user_id = -1;
            let err = missing
                .chat_append_message(&mut tokens, "user", b"hello")
                .unwrap_err();
            assert!(matches!(err, TokError::MissingToken(ref token) if token == "<|user|>"));
        }
        ModelFamily::DeepSeek4 => {
            let err = vocab
                .chat_append_message(&mut tokens, "user", b"before\0after")
                .unwrap_err();
            assert!(matches!(err, TokError::EmbeddedNul));
            let err = vocab
                .chat_append_message(&mut tokens, "user\0assistant", b"hello")
                .unwrap_err();
            assert!(matches!(err, TokError::EmbeddedNul));
        }
        ModelFamily::Glm53
        | ModelFamily::Motif3
        | ModelFamily::Dots3Note
        | ModelFamily::Qwen4Exp => {}
    }
    assert_eq!(tokens.as_slice(), &[7]);
}

#[test]
fn host_vocab_apply_matches_c() {
    let p = vocab_oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/vocab_c_oracle (missing {})",
        p.display()
    );
    let out = Command::new(&p).output().expect("run vocab_c_oracle");
    assert!(
        out.status.success(),
        "oracle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let c = String::from_utf8(out.stdout).expect("oracle utf8");
    assert_eq!(ds4_core::dump_vocab_apply_tapes(), c);
}

#[test]
fn tokenizer_families_match_c_oracle() {
    for family in [
        ModelFamily::Glm53,
        ModelFamily::DeepSeek4,
        ModelFamily::Motif3,
        ModelFamily::SolarOpen2,
        ModelFamily::ExaoneMoe,
        ModelFamily::Dots3Note,
        ModelFamily::Qwen4Exp,
    ] {
        family_cases(family);
    }
}

#[test]
fn token_type_int32_user_defined_matches_c() {
    let path = tmp("exaone-int32.gguf");
    let mut b = Builder::with_bytes().with_he();
    b.push_str("[BOS]", 3);
    b.push_str("<|endofturn|>", 3);
    b.push_str("<|system|>", 3);
    b.push_str("<|user|>", 3);
    b.push_str("<|assistant|>", 3);
    b.push_str("<|tool|>", 3);
    b.push_str("<think>", 3);
    b.push_str("</think>", 3);
    b.push_str("<tool_call>", 3);
    b.push_str("</tool_call>", 3);
    b.push_str("<tool_result>", 3);
    b.push_str("</tool_result>", 3);
    b.push_str("    ", 4);
    b.write_types(&path, "exaone-moe", 5);
    let family = ModelFamily::ExaoneMoe;
    let vocab = load(family, &path);
    assert_cmd(family, &path, &vocab, "encode", &hex_text("    hello"));
}
