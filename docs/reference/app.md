# The app

`zephyr gui` is a desktop app (Qt) that walks a project from videos to predictions. It
runs on Windows, macOS and Linux, and needs the `ui` extra.

```bash
uv run zephyr gui                          # choose or create a project
uv run zephyr gui my-project/              # a project folder; an empty one starts a project
uv run zephyr gui my-project/clips.toml    # a project, on this clip list
uv run zephyr gui my-project/clips.toml --page boxes   # straight to a page
```

## The window

- **Steps**, on the left, each marked **✓** done, **◐** partly done or **○** to do,
  from what the project folder holds. Hover a step to see what its mark rests on. A
  **●** after a step means unsaved changes.
- **Header**: the project and where it is, the **clip list** every page works on, what
  needs your attention on this page, and **Next**, highlighted once the step is done.
- **Command panel**, below the page: what the page's command is doing, with **Copy
  command**, **Cancel** and **Log**. The log opens by itself when a command fails.
- **Project → Recent projects** goes back to another project.

## Pages

| Page | Does | Command it runs |
| --- | --- | --- |
| Project | opens or creates a project; shows where each step stands | |
| 1 Clips | makes a clip list from a video folder, previewing what it would list; checks every file; **Edit TOML** and **Reload** | `clips scan`, `clips check` |
| 2 Boxes | one crop box per recording, on the list's frame size; **Auto** matches the rest | `annotate --prepare` |
| 3 Breathing | each clip's events and detector, bad stretches and clips, with suggestions for every clip; **Auto** gives every clip the default detector | |
| 4 Train | preprocesses; writes a new experiment (**Auto split** draws sessions into train, validation and test); smoke-runs, runs and resumes it | `preprocess`, `run --save-traces` |
| 5 Results | a run's scores per test group and clip, a clip's trace against its sensor, the training curves; scores a run on its test groups or another list | `run --evaluate-only`, `evaluate` |
| 6 Predict | a checkpoint's traces for a clip list or a folder of videos, with the onset probability, the crop and quality control | `clips scan --checkpoint`, `annotate --prepare`, `predict`, `quality-control run` |

## What the app writes

- It writes **clip lists** (boxes, events, marks) and **new experiments**, and never
  edits an experiment once written.
- It runs every step as a `zephyr` command in a child process on this machine, with the
  same arguments as the command line. A crash or **Cancel** never takes the app down,
  and a stopped run continues with **Resume**.
- It asks before it makes a folder a project, before it fills the feature cache (saying
  how much space it takes and how much the drive has), before it starts a run over, and
  before it replaces scores.
- Validation snippets shown while training are never saved.
- Its own settings (recent projects) live in your user settings, never in a project.

## Boxes page keys

| Key | Does |
| --- | --- |
| click | place the box, centred where you click |
| arrows | nudge by 1 pixel (++shift++: 10) |
| ++bracket-left++ / ++bracket-right++ | smaller / larger box |
| ++n++ / ++p++ | next / previous recording |
| ++1++ ++2++ ++3++ | show the recording's first, second or third clip |
| ++o++ | give this clip its own box |
| ++c++ | copy the previous recording's box |
| ++r++ | re-centre (remove the box) |
| ++a++ | Auto |

## Breathing page keys

| Key | Does |
| --- | --- |
| ++x++ | start, then end, an excluded stretch |
| ++i++ | start, then end, an included stretch (a note) |
| right-click, ++shift+x++ | delete a stretch |
| drag an edge | move a stretch |
| ++ctrl+z++ | undo |
| ++ctrl+e++ | cycle the clip's mark: unset, include, exclude |
| ++ctrl+r++ | toggle reviewed |
| ++n++ / ++p++ | next / previous clip |
| wheel, ++shift++ + wheel, drag | zoom, pan |
| ++ctrl+s++ | save |
