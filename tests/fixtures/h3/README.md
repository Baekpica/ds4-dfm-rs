# h3 media fixtures

`costello-crocodile-2panel.png` is the operator's pilot image for the
conditioning gate: a two-panel cartoon (Costello riding a crocodile, then both
in the water) to animate as a short clip.

- 432x171 PNG, 8-bit RGBA, sha256 `f364705ae974c13f709c3776ebe9e7eadcd45ada8c1a7f75d1bc7df31698f7d6`.
- Status: stored input, not yet renderable. `POST /v1/videos` refuses
  `input_reference` with 400 `unsupported_parameter`
  (`crates/ds4-core/src/minimax_h3/video_api.rs:389-410`) until P12 wires the
  first-frame and Ref2VA conditioning paths.
- Planned gate (P12): the reference takes a first frame and resizes it to the
  render canvas, so the canvas is chosen by the test, not by this file. The
  image's 2.53 aspect lands near 576x224 (2.57) and 512x192 (2.67), both legal
  canvases (edges >= 32, multiples of 32, area under 768*1344).
- The same file is the `input_reference` case for the OpenAI surface once the
  field is served; pixels arrive base64-only there.
