# Step-size pacing shared by the steppers' `run_until_cfl` loops.

from warpxm.core.types import Real


struct CflPacer:
    """Holds the current CFL step and decides when to refresh it.

        var pacer = CflPacer(dt_interval, dt_min)
        while time < t_target:
            if pacer.due():
                pacer.refresh(suggest_dt_fn(u), time)
            var remaining = t_target - time
            var dt_use = pacer.next(remaining)
            ...step by dt_use...
            if dt_use == remaining:
                return t_target
            time += dt_use
    """

    var dt: Real
    var dt_interval: Int
    var dt_min: Real
    var since_refresh: Int

    def __init__(out self, dt_interval: Int, dt_min: Real):
        self.dt = 0.0
        self.dt_interval = dt_interval
        self.dt_min = dt_min
        self.since_refresh = dt_interval

    def due(self) -> Bool:
        """True when the step should be recomputed before the next step."""
        return self.since_refresh >= self.dt_interval

    def refresh(mut self, dt: Real, time: Real) raises:
        """Adopt a new suggestion. Raises unless it is above `dt_min`
        (which also rejects NaN)."""
        if not (dt > self.dt_min):
            raise Error(
                "run_until_cfl: suggested dt " + String(dt)
                + " at t = " + String(time)
                + " is not above dt_min = " + String(self.dt_min)
            )
        self.dt = dt
        self.since_refresh = 0

    def next(mut self, remaining: Real) -> Real:
        """Size of the next step: the current `dt`, or exactly `remaining`
        when that is within round-off of it or smaller."""
        self.since_refresh += 1
        if self.dt < remaining * (Real(1.0) - Real(1.0e-12)):
            return self.dt
        return remaining
