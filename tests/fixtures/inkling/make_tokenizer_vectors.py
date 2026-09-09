#!/usr/bin/env python3
"""Generate independent vectors with the published HF tokenizer, without weights."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = args.tokenizer.read_bytes()
    config = json.loads(data)
    pipeline = json.loads(data)
    del pipeline["model"]["vocab"], pipeline["model"]["merges"], pipeline["added_tokens"]
    chat = Tokenizer.from_str(data.decode())
    config["added_tokens"] = []
    ordinary = Tokenizer.from_str(json.dumps(config))

    texts = [
        "", "hello world", "I'm I'M we've WE'VE I'd he'll IT'S", "HelloHTTPWorld",
        "123456789 ١٢٣٤٥ １２３４５ ²Ⅳ①", "한글과 English, 日本語、中文。",
        "é e\u0301 naïve ÉCOLE Straße İSTANBUL ſ's", "हिन्दी বাংলা தமிழ் العربية שלום",
        "😀👩🏽‍💻🚀\u200d✅️", "x\r\n\t  y \n \r\n", "foo/\n//bar/\r\n",
        "x\u00a0\u2002\u202f\u3000y", "x\x00\x01\x1f\x7fy",
        " a", "  a", "\t a", " \t", "  ", " \n  a", "\n\t\r  x",
        "def f(x):\n    return {'items': [1, 2, 3], '한글': True}\n",
        "<think>literal</think><|im_start|><|endoftext|>",
    ]
    added = json.loads(data)["added_tokens"]
    texts += [t["content"] for t in added]
    texts += ["x" + t["content"] + " y" for t in added]
    rng = random.Random(20260909)
    alphabet = list("AbÉéΣσİıſa\u0301\u200dع한中あह१²Ⅷ😀/!?_-' \t\n\r\u00a0\u2002\u202f\u3000")
    for _ in range(384):
        texts.append("".join(rng.choices(alphabet, k=rng.randrange(1, 96))))
    # Exercise every Unicode plane as well as common-language categories.
    for _ in range(128):
        chars = [rng.randrange(0x110000) for _ in range(16)]
        texts.append("A" + "".join(chr(c) for c in chars if not 0xD800 <= c <= 0xDFFF) + " z")

    vectors = [{"text": s, "text_ids": ordinary.encode(s).ids, "chat_ids": chat.encode(s).ids}
               for s in texts]
    args.output.write_text(json.dumps({
        "source_revision": "8cc5877b44d343f88b92086aa1fb72897950f06a",
        "tokenizer_sha256": hashlib.sha256(data).hexdigest(),
        "tokenizers_version": tokenizers.__version__,
        "pipeline": pipeline,
        "vectors": vectors,
    }, ensure_ascii=True, separators=(",", ":")) + "\n")
    print(f"{len(vectors)} vectors -> {args.output}")


if __name__ == "__main__":
    main()
