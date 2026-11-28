# ZeroFG source snapshot

This directory contains the engine of ZeroFG **v1.2.1** (`c5e9b64`,
https://github.com/hy300leosquizz-ctrl/ZeroFG/releases/tag/v1.2.1), on top of the MFG branch `mfg/core` at `4f761e8` (1.2g, the
HUD layer at every phase), itself on top of the engine of **v1.2.0** (`3376106`,
https://github.com/hy300leosquizz-ctrl/ZeroFG/releases/tag/v1.2.0). It sits on top of v1.1.0 and the license commit `10e68b6`.

Over v1.2.0, 1.2.1 adds two RZ+ product options (the public API is unchanged):
- the flow guided by colour at motion edges (`DF_PRODUCT_OPTS 256`): the coarse flow is upsampled by its own picture's colour, so
  the square blocks of motion at an edge round off;
- the **Advanced VTF Filter** (Vector Trail Fill, `DF_PRODUCT_OPTS 512`), the occlusion halo disguised as the background's motion: beside a character the camera carries, the
  cleanup's blur becomes a fill fetched from both Reals along one elected background motion (a new pass, `df_halo`, decides per
  4 x 4 block), clamped to the colours already around it.

ZeroFG is `Apache-2.0 OR GPL-2.0-or-later` since that
commit; it is used here under Apache-2.0, `LICENSE`.

The earlier history: over v1.1.0, v1.2.0 adds:
- any phase between the Reals (x3 / x4), with the same-pair contract;
- Zero's screen by cells (`kHudCells`);
- Zero's light gain (`kLightGain`);
- Zero's stop rule (`kStopStill`): when the camera stops, the fallback surface holds still instead of carrying the last search's
  motion; and a switch for the base's cleanup (`kBaseClean`, kept on);
- the render's two passes in one walk (the same picture), and `CreateInfo::zero_options`: `kZeroLight` (one background surface, one
  render pass, no guided filter) and `kZeroQuarterBase` (x4's quarter phases from the base's picture), both off by default in the engine.
- a wipe interpolates (`kCutLit`): a pair is a cut (B whole) only if the points lit in both pictures do not correlate either.
- fast motion is no cut (`kCutCost`): a cut also needs the motion search to have failed nearly everywhere.
- a calm pair is the base's (`kEasy`): where the motion search explains nearly everything, the base's picture whole.
- ReallyZero becomes RZ+ (2026-11-20): the camera's shift (one exhaustive search on the coarsest level) is a start of the patch search,
  and the HUD hold keeps what stayed in place with texture for six pairs in place; a cut also needs the points lit in both pictures
  not to correlate (a wipe interpolates, as Zero's kCutLit).

At the middle, with the switches off (`kHudCells`, `kLightGain`, `kStopStill` 0; `kBaseClean` 1), it is bit-identical to v1.1.0, except on the pairs the old cut rule took for cuts and `kCutLit` or `kCutCost` do not. The engine sources, public headers, shader sources, shipped SPIR-V
headers, shader compiler and CMake file are those of `c5e9b64` (v1.2.1), byte for byte. The snapshots before it were `4edb614` (the 1.2.1
test build, the same code), `3376106` (v1.2.0) and `f9b0b5cef` (v1.1.0).

No laboratory recordings, runtime binaries, device identifiers or experimental environment controls are included.

The host adapter in `integration/xenia/` is the thin adapter from the same
release's `host/xendroid/` directory. Xendroid's presenter owns recording order,
queue submission, synchronization, presentation and resource lifetimes.

Product mapping in this integration:

| User choice | Core mode |
|---|---|
| Off | Native Xendroid+ presentation |
| On, x2 / x3 / x4 | `zerofg::Mode::kReallyZero` (RZ+, the engine of 1.2.1), N - 1 frames between two of the game's |

The layered engine (`zerofg::Mode::kZero`, RC7) is still in the snapshot but no longer offered: RZ+ matches or beats it at a
fraction of the cost. The integration names the modes `Mode::kDenseFlow` and `Mode::kRC7`, their development names. One interpolator serves the selected mode, with
per-flight descriptors. Recordings are submitted serially in their recording
order; an abandoned recording cannot be followed by a retry using its retained
history. Mode changes drain the old session before recreating generation state.

The engine is Apache-2.0; see `LICENSE`. Xenia-derived host integration files
outside this directory retain their original notices.
