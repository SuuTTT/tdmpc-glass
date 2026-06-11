"""Synthetic / controlled environments for mechanism-check experiments.

These are NOT the MuJoCo Playground training tasks (those live behind
``scripts/run_benchmark.py`` on GPU workers). The modules here are small,
pure-JAX worlds whose *ground-truth* structure is known by construction, so a
later probe can be validated against it.
"""
