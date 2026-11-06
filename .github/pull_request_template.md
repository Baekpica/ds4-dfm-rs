## Change

<!-- What changed and why. One logical change per commit. -->

## Gates

<!-- Each command run, its result and the host (GPU, clocks, artifact hashes).
     Say what is unverified. -->

## Checklist

See [CONTRIBUTING.md](../CONTRIBUTING.md#model-families-and-large-contributions).

- [ ] No one-off experiments, launcher or per-machine scripts, host paths or
      references to files outside this repository
- [ ] Serving uses the common `ds4-server` / `ds4` / `ds4-bench` commands;
      family switches are `DS4_<FAMILY>_*` variables documented on the
      family page
- [ ] `serving_caps` declarations, defaults, README rows and docs match the
      recorded evidence; generated docs regenerated
- [ ] Bug fixes come with a test that failed before the fix
- [ ] `make ci-lint`, `make ci-rust` and the family gate pass; no new
      compiler warnings against main
