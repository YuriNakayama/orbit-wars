"""Shared JAX-native core for the faithful baseline_v1 port.

Vectorized, jit/vmap-friendly reimplementations of baseline/core/* that match
the Python originals within float32 tolerance (x64 exact). Bottom-up layers:
geometry → physics → safety → worldmodel → missions.
"""
