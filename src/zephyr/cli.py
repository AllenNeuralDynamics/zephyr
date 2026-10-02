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
    zephyr annotate ...
    zephyr preprocess ...
    zephyr train ...
    zephyr evaluate ...
    zephyr benchmark plan|run|collect ...
    zephyr benchmark-data download|validate ...
    zephyr benchmark-report head|report|all ...
    zephyr baseline pixel|facemap|net|collect ...
"""

from pydantic import BaseModel
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


class AnnotateCommand(_PassthroughCommand):
    """Cache frames and place hand crop boxes. See ``zephyr.annotate``."""

    def cli_cmd(self) -> None:
        from . import annotate

        annotate.main(self.args)


class PreprocessCommand(_PassthroughCommand):
    """Decode, crop, and cache channel arrays. See ``zephyr.preprocess``."""

    def cli_cmd(self) -> None:
        from . import preprocess

        preprocess.main(self.args)


class TrainCommand(_PassthroughCommand):
    """Train a CNN-TCN checkpoint. See ``zephyr.train``."""

    def cli_cmd(self) -> None:
        from . import train

        train.main(self.args)


class EvaluateCommand(_PassthroughCommand):
    """Score a checkpoint on held-out clips. See ``zephyr.evaluate``."""

    def cli_cmd(self) -> None:
        from . import evaluate

        evaluate.main(self.args)


class BenchmarkCommand(_PassthroughCommand):
    """Run the factorial train/test benchmark sweep. See ``zephyr.benchmark``.

    Takes its own ``plan|run|collect`` sub-subcommand, e.g.
    ``zephyr benchmark run --dry-run``.
    """

    def cli_cmd(self) -> None:
        from . import benchmark

        benchmark.main(self.args)


class BenchmarkDataCommand(_PassthroughCommand):
    """Download and validate the benchmark dataset. See ``zephyr.benchmark_data``.

    Takes its own ``download|validate`` sub-subcommand.
    """

    def cli_cmd(self) -> None:
        from . import benchmark_data

        benchmark_data.main(self.args)


class BenchmarkReportCommand(_PassthroughCommand):
    """Build the benchmark's final plots and summary. See ``zephyr.benchmark_report``.

    Takes its own ``head|report|all`` sub-subcommand.
    """

    def cli_cmd(self) -> None:
        from . import benchmark_report

        benchmark_report.main(self.args)


class BaselineCommand(_PassthroughCommand):
    """Run and collect the manuscript baselines. See ``zephyr.baseline``.

    Takes its own ``pixel|facemap|net|collect`` sub-subcommand.
    """

    def cli_cmd(self) -> None:
        from . import baseline

        baseline.main(self.args)


class Zephyr(BaseSettings):
    """Zephyr: CNN + TCN network for predicting breathing from video."""

    model_config = SettingsConfigDict(cli_kebab_case=True)

    annotate: CliSubCommand[AnnotateCommand]
    preprocess: CliSubCommand[PreprocessCommand]
    train: CliSubCommand[TrainCommand]
    evaluate: CliSubCommand[EvaluateCommand]
    benchmark: CliSubCommand[BenchmarkCommand]
    benchmark_data: CliSubCommand[BenchmarkDataCommand]
    benchmark_report: CliSubCommand[BenchmarkReportCommand]
    baseline: CliSubCommand[BaselineCommand]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def run() -> None:
    CliApp.run(Zephyr)


if __name__ == "__main__":
    run()
