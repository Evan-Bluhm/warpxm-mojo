# 3D shim around the consolidated Euler app. See `apps/euler.mojo` for
# the canonical implementation.

from warpxm.apps.euler import Euler

comptime Euler3D = Euler
