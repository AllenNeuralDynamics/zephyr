# Working in zephyr

Rules for coding agents. Behaviour is documented in the READMEs, `docs/` and the pydantic
models; this file only says what must not be done.

## Code

- No dataset specifics in `src/` (paths, split names, cameras, sessions, layout).
- Never weaken a loading or `evaluate` check to make a config pass; fix the config.
- Paths that identify a clip go through `config.canonical`, never `str()` or a raw
  comparison.
- Never guess a frame rate from a video stream; a video with no timestamps is given one.
- Feature caches are read only through `features.reader`.
- Code never rewrites existing checkpoints, `config.json` / `args.json`, histories or
  traces. `data/` is read-only.
- Keep the `__path__` extension in `zephyr/__init__.py`; never add an `__init__.py` at
  `packages/*/src/zephyr/`.

## The app (`zephyr.gui`)

- It only runs `zephyr` commands (`gui/worker.py`'s `COMMANDS`) in a local child
  process. A step it needs that no command does is a new command first.
- It writes only clip lists and new experiments, never an existing experiment or a run
  folder outside its project.
- It never silently re-scores, restarts a run, creates a project or fills the cache: the
  user agrees first.
- Qt is imported only by `gui/`'s window modules; logic lives in headless classes.
- Never drop a key or control of the Boxes or Breathing pages (`tests/test_gui_qt.py`).


## Browser app (`web/`)

- Zephyr never imports it. Never commit model files; they are release assets listed in
  `web/models.json`.
- A change to what `zephyr predict --video` does is a change here too.

## Making a change

- New behaviour is a new pydantic field with a default that keeps current behaviour.
- Add a test, then run `uv run ruff format . && uv run ruff check . && uv run pytest -q`
  from the root.
- Update the relevant README, `docs/` page and this file in the same change. Screenshots
  and the demo come only from `scripts/docs_screenshots.py` and `scripts/demo_media.sh`,
  never mocked or hand-edited.
