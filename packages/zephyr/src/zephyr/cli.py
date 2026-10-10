"""Single ``zephyr`` entry point, dispatching to each module's own CLI.

Every subcommand below is a thin passthrough: it only recognizes its own name
and forwards every other argument, unparsed, to that module's existing
``argparse``-based ``main()``. Each module keeps defining and validating its
own flags -- this file's only job is picking which module runs.
``zephyr <subcommand> --help`` only prints this docstring's one-line summary,
not the module's full flag reference -- see that module's own docstring for
the complete list.

CLI
---
    zephyr clips scan <dir> --glob <pattern> -o <list.toml>
    zephyr annotate <list.toml> [--prepare]
    zephyr annotate events [<list.toml>]
    zephyr preprocess <list.toml>... --cache <dir>
    zephyr run <experiment.toml> [--smoke] [--folds <name>...]
    zephyr evaluate --checkpoint <best.pt> --clips <list.toml>... --cache <dir>
    zephyr schema <dir>
"""

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    CliApp,
    CliSubCommand,
    CliUnknownArgs,
    SettingsConfigDict,
)


class _PassthroughCommand(BaseModel):
    """Forwards every argument, unparsed, to the wrapped module's ``main()``."""

    args: CliUnknownArgs = []


class ClipsCommand(_PassthroughCommand):
    """Create clip lists. See ``zephyr.clips``."""

    def cli_cmd(self) -> None:
        from . import clips

        clips.main(self.args)


class AnnotateCommand(_PassthroughCommand):
    """Place crop boxes, or (``annotate events``) choose each clip's event detector."""

    def cli_cmd(self) -> None:
        from . import annotate

        annotate.main(self.args)


class PreprocessCommand(_PassthroughCommand):
    """Decode, crop, and cache channel arrays. See ``zephyr.preprocess``."""

    def cli_cmd(self) -> None:
        from . import preprocess

        preprocess.main(self.args)


class RunCommand(_PassthroughCommand):
    """Train and evaluate a fold or an experiment. See ``zephyr.run``."""

    def cli_cmd(self) -> None:
        from . import run

        run.main(self.args)


class EvaluateCommand(_PassthroughCommand):
    """Score a checkpoint on held-out clips. See ``zephyr.evaluate``."""

    def cli_cmd(self) -> None:
        from . import evaluate

        evaluate.main(self.args)


class SchemaCommand(_PassthroughCommand):
    """Write JSON schemas of the config files. See ``zephyr.config``."""

    def cli_cmd(self) -> None:
        from . import config

        config.main(self.args)


class Zephyr(BaseSettings):
    """Zephyr: CNN + TCN network for predicting breathing from video."""

    model_config = SettingsConfigDict(cli_kebab_case=True)

    clips: CliSubCommand[ClipsCommand]
    annotate: CliSubCommand[AnnotateCommand]
    preprocess: CliSubCommand[PreprocessCommand]
    run: CliSubCommand[RunCommand]
    evaluate: CliSubCommand[EvaluateCommand]
    schema_: CliSubCommand[SchemaCommand] = Field(alias="schema")

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def run() -> None:
    CliApp.run(Zephyr)


if __name__ == "__main__":
    run()
