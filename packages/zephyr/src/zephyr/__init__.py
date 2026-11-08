"""Breathing model, data preparation, training, and inference utilities."""

# zephyr-manuscript's zephyr-benchmarks and zephyr-dataset install zephyr.benchmarks
# and zephyr.corpus from separate distributions; extending the path lets them live
# under this import name.
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
