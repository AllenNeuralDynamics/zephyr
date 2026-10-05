"""Breathing model, data preparation, training, and inference utilities."""

# zephyr-benchmarks installs zephyr.benchmarks from a separate distribution;
# extending the path lets both live under one import name.
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
