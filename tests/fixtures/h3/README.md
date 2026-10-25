# h3 media fixtures

`costello-crocodile-2panel.png` is the operator's pilot image for the
conditioning gate: a two-panel cartoon (Costello riding a crocodile, then both
in the water) to animate as a short clip.

- 432x171 PNG, 8-bit RGBA, sha256 `f364705ae974c13f709c3776ebe9e7eadcd45ada8c1a7f75d1bc7df31698f7d6`.
- Status: a first-frame anchor. `POST /v1/videos` takes it as the clip's first
  frame, as a base64 `data:` URI or a multipart file part
  (`crates/ds4-core/src/minimax_h3/video_api.rs:526-592`,
  `crates/ds4-server/src/h3_video.rs:794-820`); the `h3-rs` CLI takes it from
  disk with `--first-frame`. Ref2VA ordered references remain unported.
- Measured bound (P12.9): the 16x VAE reproduces a large enough anchor, but this
  page's lettering is ~4 px glyphs and does not survive downscaling into a
  legal canvas; a re-lettered 1536x608 page round-trips at 28.90 dB.
- Anchor geometry: the reference takes a first frame and resizes it to the
  render canvas, so the canvas is chosen by the test, not by this file. The
  image's 2.53 aspect lands near 576x224 (2.57) and 512x192 (2.67), both legal
  canvases (edges >= 32, multiples of 32, area under 768*1344).
- The same file is the `input_reference` case for the OpenAI surface; pixels
  arrive base64-only there.
