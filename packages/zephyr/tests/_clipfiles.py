"""Helpers that lay out a tiny fake dataset and write clip lists/folds for tests."""

from pathlib import Path


def touch_clip(
    folder: Path, group: int, part: int, *, camera: str = "face", thermistor=True
) -> Path:
    """Create the files of one clip; return the video path."""
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"video_{camera}_{group}_part_{part}"
    (folder / f"{stem}.mp4").touch()
    (folder / f"{stem}.parquet").touch()
    if thermistor:
        (folder / f"thermistor_{group}_part_{part}.parquet").touch()
    return folder / f"{stem}.mp4"


def _rel(target: Path, base: Path) -> str:
    return Path(target).relative_to(base, walk_up=True).as_posix()


def write_list(
    path: Path,
    videos: list[Path],
    *,
    box=(10, 10, 32, 32),
    target_size=(100, 80),
    extra: str = "",
) -> Path:
    """Write a clip list naming *videos* (relative to *path*) with one box each."""
    lines = [f"[preprocess]\ntarget_size = {list(target_size)}\n{extra}\n"]
    for video in videos:
        lines.append(
            f'[[clip]]\nvideo = "{_rel(video, path.parent)}"\nbox = {list(box)}\n'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def write_fold(
    path: Path, train: list[tuple[Path, float]], test: dict[str, Path]
) -> Path:
    lines = [
        f'[[train]]\nclips = "{_rel(clips, path.parent)}"\nweight = {weight}\n'
        for clips, weight in train
    ]
    lines.append("[test]")
    lines += [f'{name} = "{_rel(clips, path.parent)}"' for name, clips in test.items()]
    path.write_text("\n".join(lines) + "\n")
    return path
