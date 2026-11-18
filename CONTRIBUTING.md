# Contributing to DUET

Thanks for looking. A few ground rules:

- **Licence.** DUET is under FSL-1.1-ALv2 (see `LICENSE`). We don't accept code contributions from outside GRAPHIQ
  Studio yet, because they would need a contributor agreement first. Issues and ideas are welcome.
- **No voices of real people.** Supertonic-3's OpenRAIL-M licence forbids impersonation. Design voices, never clone
  one, and don't add recordings of anyone who hasn't agreed.
- **No models or recordings in git.** `models/` and `runs/` stay out of the repository; name any model you add in
  `NOTICE` with its licence.
- **Measure before you claim.** Every number in the README has its run in `docs/DETAILS.md`. Don't report a latency
  or an accuracy you didn't measure on named hardware.
- **Before a pull request:** every file in `tests/` passes (each runs on its own, `python tests/test_aec.py`, or all
  with pytest), and `python -m duet sim front-desk` still answers in time.
