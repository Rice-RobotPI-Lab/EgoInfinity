# media/

Figures and videos for the EgoInfinity project page. Use the exact filenames
below: each maps to a placeholder in `index.html`, which gets swapped for a
real `<img>`/`<video>` once the file is present.

## Teaser / overview
- `teaser.mp4`            top teaser loop (or a hero still image)

## Pipeline  (section 02)
- `pipeline.jpg`          engine schematic, paper Fig. 1

## Retargeting  (section 03)
- `retarget.jpg`          root-frame estimator, paper Fig. 2 / 7

## Experiments  (section 04) — real dual-arm Franka FR3 skills
- `cut.mp4`               (labelled "Cut")
- `pour_bowl.mp4`         (labelled "Pour bowl")
- `pour_glass.mp4`        (labelled "Pour glass")
- `swipe_box.mp4`         (labelled "Wipe box")
- `swipe_computer.mp4`    (labelled "Wipe computer")

## Experiments — LEAP-hand grasping policy
- `pick.mp4`              one wide strip; left to right: apple, banana, tomato can

## Notes
- Web-optimize videos: H.264, ~720p, `-movflags +faststart`, `-an` (muted autoplay).
- After adding files here, the placeholders in index.html get wired to `media/<file>`.
