"""hypothesis_testing_arena: six group-difference tests judged by simulation.

The library is deliberately split three ways:

* ``src.tests``      - the six tests, each written from scratch and validated against scipy.
* ``src.simulation`` - populations and scenarios that carry a *known* ground truth.
* ``src.evaluation`` - the scoring: false-positive rate, power, p-value calibration, the arena.

Nothing here reads a dataset. Every number in the project comes from simulation seeded from
``config/settings.yaml``.
"""

__version__ = "0.1.0"
