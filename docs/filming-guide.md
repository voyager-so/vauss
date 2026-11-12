# How to film a room (about 1 minute on a phone)

A 3D room is built from how near and far things shift against each other as the camera
moves. Turning on the spot shows no depth, and fast or dark video comes out smeared.
`vauss run` checks every video before anything is uploaded; `vauss check <video> --room`
runs the same check on its own in a few seconds. Each line of its report is named after a
section below (Length, Light, Sharpness, Movement, Detail, Speed, Coverage).

## Before you press record

- **Lights:** turn on every light and open the curtains. More light means a faster shutter
  and sharper frames. Keep direct sun out of the shot.
- **Clear the room** of people and pets. Anything that moves turns into a ghost.
- **Camera:** the main 1x lens (not the 0.5x ultra-wide), 4K or 1080p, landscape or
  portrait. Tap a mid-bright wall to focus. HDR video (the iPhone and Pixel default) is fine:
  the check judges it as it looks on screen, so it reads the same as an SDR copy.

## The path: walk arcs, don't spin

```
   DON'T: stand still and spin            DO: walk a slow arc in front of each wall

   +------------------------+             +------------------------+
   |                        |             |   wall A               |
   |                        |             |    ^   ^   ^   ^       |
   |           @            |             |  (------------->)      |  1-2 m from
   |         <-+->          |             |    walk sideways,      |  the wall
   |                        |             |    facing the wall     |
   |   no depth: the camera |             |                        |
   |   never moves          |             |  then turn to wall B,  |
   +------------------------+             |  C, D and do the same  |
                                          +------------------------+
```

1. Stand **1-2 m from a wall**, facing it. Walk a slow sideways arc 1-2 m across while
   keeping the wall in view. Then turn to the next wall and repeat, all the way round.
2. **Twice:** once at eye height, once at chest height with the phone tilted down about
   30 degrees (this catches the floor and the furniture tops).
3. Give **every wall at least 4 seconds**, including bare walls, doors and corners. On a
   plain wall, keep an edge, a door frame or a picture in view.
4. **Slow:** about one step a second, and turn at **10 degrees a second or less**: a full
   turn of the room takes 40 seconds or more (the check estimates the blur this causes; more light
   lets you turn a little faster). One take of about 1 minute, no cuts or pauses (longer clips only take longer to process).
   Hold the phone with both hands, elbows in.

## What the check's report means

A **FAIL** stops `vauss run` before anything is uploaded (`--ignore-check` runs it anyway).
A **WARN** still runs, but only shows when our tests found that problem goes with a worse
result: the room comes out softer or patchier, or the run may fail. A **TIP** is optional advice:
these did not make results measurably worse in our tests, but fixing them is still good practice.
Times like `0:22-0:30` are positions in your video.

### Length
- FAIL `Too short: 2.0 s.`: under 3 seconds.
- TIP `Short: 17 s.`: under 30 s for a room (10 s for an object or place).
- WARN `Long: 120 s. 1 minute is enough; a longer clip only takes longer to process.`: over 75 s.
- Fix: film for about 1 minute, a slow loop of the whole room.

### Light
- FAIL `Almost black: median brightness 3/255.`: the video is essentially black.
- TIP `Dim: 85% of the video is dark (median brightness 40/255; aim for 100+).`: median
  brightness under 64/255, or more than half the video dim.
- Fix: turn on every light and open the curtains; dim video blurs and gets noisy.

### Sharpness
- WARN `Blurry: 41% of frames are much softer than the rest`: more than 35% of frames
  under half the video's typical sharpness (shake, focus hunting).
- WARN `Blurry in places: 0:57-1:08 mostly blurred frames.`: stretches of 5 s or more where
  most frames are blurred, adding up to 10 s or more. Shorter blurred stretches are a TIP.
- TIP `Smeared by motion: 0:03-0:08 ~21 px, ...; more light or a slower turn.`: a stretch
  of 5 s or more where each frame is smeared more than 15 px (measured at 1080p) by the
  camera turning during the exposure. In our room tests, views filmed with that much smear
  rendered about a third less sharp. Dim rooms make it worse: the phone keeps the shutter
  open longer.
- Fix: hold the phone steady with both hands, tap the screen to focus, add light, and turn
  slower (10 degrees a second or less).

### Movement
- FAIL `The camera barely moved (about 0.2 image widths, almost no depth change between
  views)`: the view shifted less than one image width in total, and even the best pair of
  views shows almost no depth change. There is nothing to build 3D from.
- TIP `Mostly turning in place: little depth change between views`: the camera turned
  more than it moved.
- TIP `Little camera movement ... views too different to measure depth change.`
- Fix: walk, don't spin. Walk a slow arc 1-2 m across in front of each wall, at two heights.

### Detail
3D is built by matching the same points (corners, edges, texture) between frames. A plain wall,
a blurry or a dark video gives too few of them.
- WARN `Too little detail to match: about 10 distinct points a frame and 1 matched between
  frames (plain walls, blur or darkness); the views may not tie together, so this video will
  probably fail.`: about 50 distinct points a frame or fewer, and 5 or fewer matched between
  frames. Re-filming is the better bet.
- WARN `Little detail to match: about 9 points matched between frames (aim for 20+).`: 12 or
  fewer matched between frames.
- Fix: keep textured things in view (furniture, shelves, pictures, door frames), stay 1-2 m
  from plain walls, turn slower and add light.

### Speed
Turning blurs each frame by about (turn speed × the time the shutter is open). In a dim room the
phone keeps the shutter open for the whole frame, so the same turn blurs twice as much as in a
bright one.
- FAIL `Too fast: frames stop overlapping for 7 s at a time (a hyperlapse or very fast
  pans).`: 5 s or more where each frame shares nothing with the one before.
- WARN `Turning too fast for the light: 0:05-0:13 ~16 px of blur a frame, turning ~45 degrees
  a second; turn slower or add more light.`: a stretch of 5 s or more (or most of the video)
  with about 12 px of blur a frame or more.
- TIP `Move a bit slower for sharper results (about 9 px of blur at 0:12-0:17); more light
  also helps.`: about 8-12 px of blur a frame. The video still runs; it would just be sharper.
- TIP `Jumpy: 30% of frames don't overlap the one before.`: more than a quarter.
- Fix: turn slower (about 10 degrees a second, a full turn in 40 s or more) or add light.

### Coverage (rooms)
Directions are measured from where the camera first pointed: "45 degrees right", "your
right (90)", "behind you", and so on.
- TIP `Part of the room is missing: never filmed toward behind you, ... (turned about 150 of
  360 degrees).`: the camera turned less than 180 degrees in total, so about half the room
  was never in view.
- A passing line can still note `quick glances only: your left (90) (2 s), ...`: directions
  filmed for under 4 seconds. Worth an extra slow pass if those walls matter to you; the
  estimate of where the camera pointed is rough, so it's a hint, not a warning.
- Fix: point the camera at every wall, the corners and the floor. Give each wall at least
  4 s, with extra slow passes on bare walls.
