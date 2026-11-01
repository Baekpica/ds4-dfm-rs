# The Gemma4 tokenizer: port map from the reference implementation (2026-10-30)

Unit 3's source read. The asset lives inside the checkpoint as two U8 blobs -
`tokenizer_json` (32,169,626 B) and `hf_asset__tokenizer_config.json`
(2,749 B) - so nothing here needs a file from outside it. The wrapper the
release calls is its
`text_encoders/gemma/tokenizer.py`, and the implementation under it is
HuggingFace's `tokenizers`, which the pinned Python wraps. `transformers`
5.14.1 requires `tokenizers>=0.22.0,<=0.23.0`
(`transformers-5.14.1.dist-info/METADATA`); 0.23.0 is not on crates.io, so the
source read is **0.22.1**, fetched from crates.io and unpacked read-only at
`/data/refs/tokenizers/tokenizers-0.22.1`.

The operator's decision (2026-10-30) is the repo's own BPE rather than the
`tokenizers` crate, so the reference is read to be re-implemented, and the
deviation is named in section 6.

## 1. What the JSON states

    version 1.0, truncation null, padding null
    normalizer     Replace   pattern {"String": " "}   content "▁"
    pre_tokenizer  Split     pattern {"String": " "}   behavior MergedWithPrevious
    post_processor TemplateProcessing, single = [Sequence A], special_tokens {}
    decoder        Sequence [Replace "▁"->" ", ByteFallback, Fuse]
    model          BPE, unk_token "<unk>", fuse_unk true, byte_fallback true,
                   ignore_merges false, continuing_subword_prefix null,
                   end_of_word_suffix null, dropout null
    added_tokens   <pad> 0, <eos> 1, <bos> 2, <unk> 3, <mask> 4, ...: all
                   single_word/lstrip/rstrip false, normalized false, special true
    vocab          262,144 entries; merges 514,905 pairs

Three consequences the port depends on:

- this is a **SentencePiece-style** vocabulary, not a GPT-2 byte-level one:
  the alphabet is characters, the space is `▁` (U+2581), and the byte codec has
  no part in it;
- the `normalizer` turns every space into `▁` *before* the pre-tokenizer, which
  then splits on a literal `" "` - so the split finds nothing and the whole
  normalised text is one piece (section 3);
- the post-processor adds **no** special tokens, which is why the wrapper
  prepends the BOS itself.

## 2. Added tokens first, on the raw text

`tokenizer/added_vocabulary.rs:462-465`: tokens whose `normalized` is false are
split out of the **non-normalized** string before the normalizer runs, and the
gaps between them are normalised, pre-tokenized and BPE'd. `find_matches`
(`:367-...`) walks the added-token automaton left to right; `single_word`,
`lstrip` and `rstrip` are all false in this asset, so those branches never
fire, and `encode_special_tokens` with a special token in the set leaves the
token to the model instead of the wrapper.

## 3. Normalize, then pre-tokenize

- `normalizers/replace.rs:82-86` is `NormalizedString::replace(pattern, content)`
  over a **literal** pattern here: every `" "` becomes `"▁"`.
- `pre_tokenizers/split.rs:141-149` documents `MergedWithPrevious`: the
  delimiter goes on the **end of the preceding** piece (`"How are"` ->
  `["How ", "are "]`). Since section 3's normalizer has already removed every
  space, the split yields exactly one piece - the whole text. The port asserts
  that rather than assuming it: a normalised string containing `" "` is refused.

## 4. The model

`models/bpe/model.rs:382-466`, `merge_word`:

1. walk the piece by **characters** (`char_indices`), not bytes;
2. for each character, look it up in the vocab;
3. on a miss, if `byte_fallback`, map each of the character's UTF-8 bytes to
   `format!("<{b:#04X}>")` - **uppercase** hex - and use those ids if *all* of
   them are in the vocab (`:417-431`);
4. otherwise the character joins the pending `unk`; with `fuse_unk` true a run
   of unknown characters collapses into one `<unk>` id (`:437-440`);
5. then `Word::merge_all`.

`merge_all` (`models/bpe/word.rs:162-...`) is a min-heap of adjacent pairs keyed
by their merge rank, and `Merge`'s ordering (`word.rs:26-35`) is rank first and
position second - so the **lowest rank wins and the leftmost position breaks a
tie**, one occurrence at a time, with stale entries discarded and the pairs
either side of a merge re-pushed. A scan that repeatedly finds the lowest-rank
adjacent pair, leftmost on a tie, and merges that single occurrence is the same
algorithm; the heap is only the ordering device.

Note the byte fallback's reach in this asset: the vocab carries only the 100
symbols `<0x00>` .. `<0x99>`, whose two hex digits are both in `0`-`9`. So
`<0x9A>`, `<0xA0>`, `<0xFF>` and 156 others are absent, and for a byte whose
uppercase spelling is missing the `Option<Vec<_>>` at `:419-427` is `None` and
the whole **character** becomes `<unk>` instead. A port that minted
`<0x9A>` on demand would silently differ.

## 5. The wrapper

`tokenizer.py:39-63`:

1. `text.strip()`;
2. encode with `truncation=True, max_length=1024` - so the ids are truncated to
   1024 **before** the BOS is considered;
3. prepend `bos_token_id` (2) unless the first id already is it, and truncate to
   1024 again (`[bos, *ids][:1024]`);
4. left-pad to 1024 with `pad_token_id` (0), `padding_side = "left"` from
   `tokenizer_config.json:20`;
5. return `(id, attention_mask)` pairs; the mask is the second element
   `encoders/base_encoder.py:56-66` forwards.

The `post_processor` adds nothing, so the ids here are the model's ids with the
BOS the wrapper prepends - which is why `tokenizer_config.json` also has to be
read rather than only the JSON.

## 6. Named deviations

- **The `tokenizers` crate is not used** (operator decision). This is a
  re-implementation of the same algorithm, so it is a deviation by
  construction and is measured against the golden's ids rather than assumed
  equivalent.
- The added-token automaton is modelled as a leftmost, longest-on-ties literal
  scan over the added-token contents; the reference searches a regex set
  (`find_matches`), and the two agree unless two added tokens overlap, which
  none in this asset do.
- `serde_json` parses the 32 MB blob; the reference parses the same JSON with
  serde as well, so this is not a deviation at all - recorded to say so.

## 7. What unit 3 can and cannot gate

There is no reference tokenizer on either host: neither has Python with
`tokenizers`, and the operator's decision keeps the crate out of the workspace.
So the gate is:

- a structural check of the JSON against what the port accepts, refusing any
  other pipeline shape by name;
- unit tests over the merge order, the byte fallback's reach, the `▁`
  normalisation, the added-token split and the wrapper's pad/truncate order;
- **the golden's ids and mask**, which are the operator's (handoff section 3c)
  and are the only external statement of what the reference produces.
