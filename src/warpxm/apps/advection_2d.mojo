# 2D shim around the consolidated Advection app. See `apps/advection.mojo`.

from warpxm.apps.advection import Advection

comptime Advection2D = Advection
