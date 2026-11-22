"""Pure protocol checks, independent of DDS or simulation rendering."""

import math

VELOCITIES = {"LEFT": (0.18, 1.5), "STRAIGHT": (0.35, 0.0), "RIGHT": (0.18, -1.5)}


def stamp_parts(timestamp):
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("Temps invalide")
    nanos = round(timestamp * 1_000_000_000)
    return divmod(nanos, 1_000_000_000)


def validate_velocity(label, linear, angular):
    if label not in VELOCITIES or not math.isfinite(linear) or not math.isfinite(angular):
        raise ValueError("Commande invalide")
    expected = VELOCITIES[label]
    if not math.isclose(linear, expected[0], abs_tol=1e-9) or not math.isclose(angular, expected[1], abs_tol=1e-9):
        raise ValueError("Vitesse incoherente avec la decision")


class DecisionGate:
    def __init__(self, run_id):
        self.run_id = run_id
        self.expected = None
        self.last_window = 0

    def begin(self, window_id, stamp):
        if self.expected is not None or window_id <= self.last_window:
            raise ValueError("Fenetre deja en attente ou identifiant non croissant")
        self.expected = (window_id, tuple(stamp))
        self.last_window = window_id

    def accept(self, run_id, window_id, stamp):
        if run_id != self.run_id or self.expected != (window_id, tuple(stamp)):
            return False
        self.expected = None
        return True
