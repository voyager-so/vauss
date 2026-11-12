# Spirula: clean-room statement

[Spirula Studio](https://github.com/harry7557558/spirula-studio) is licensed GPL-3.0. splat contains no Spirula code.

- **Reference only.** Contributors read Spirula's source and design notes to understand how it handles hard
  phone-video inputs (frame selection, SfM matching and mapping edge cases). Every behaviour splat adopted was
  re-described in our own words and implemented from that description. No Spirula code was quoted, paraphrased or
  translated into this repository.
- **Optional subprocess only.** The optional `spirula` engine (`crates/splat-spirula`) runs an unmodified Spirula
  release binary as a separate process. splat never links, vendors, embeds or redistributes it. The binary is
  downloaded only when a user asks for it explicitly (`splat fetch spirula --allow-gpl`) and is never included in a
  splat release archive.
- Details of the subprocess boundary and dependencies: [`crates/splat-spirula/DEPENDENCIES.md`](../../crates/splat-spirula/DEPENDENCIES.md).
