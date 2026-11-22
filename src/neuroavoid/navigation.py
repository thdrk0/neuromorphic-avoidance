"""Navigation task and privileged map-based reference controller."""

from dataclasses import dataclass
import heapq
import math

import mujoco
import numpy as np


@dataclass(frozen=True)
class NavigationTask:
    goal: tuple = (2.2, 0.0)
    goal_radius: float = 0.28
    timeout: float = 60.0

    def outcome(self, sim):
        # A simultaneous goal contact must never count as a collision-free success.
        if sim.collided:
            return "collision"
        if math.dist(sim.pose[:2], self.goal) <= self.goal_radius:
            return "success"
        if sim.data.time + 1e-9 >= self.timeout:
            return "timeout"
        return "running"


class ReferenceController:
    """A* on inflated static obstacles, followed using LEFT/STRAIGHT/RIGHT.

    Uses simulator geometry and true robot pose; never a visual-policy baseline.
    The clearance covers the robot footprint and steering arcs in this scene.
    """

    clearance = 0.45
    spacing = 0.15

    def __init__(self, sim, task):
        self.goal = task.goal
        self.obstacles = []
        for i in range(sim.model.ngeom):
            name = sim.model.geom(i).name
            if name.startswith(("wall_", "obstacle_")):
                self.obstacles.append((int(sim.model.geom_type[i]),
                                       sim.model.geom_pos[i, :2].copy(),
                                       sim.model.geom_size[i].copy()))
        self.path = self.plan(sim.pose[:2], task.goal)
        self.target = 1

    def free(self, point):
        x, y = point
        if abs(x) > 2.5 or abs(y) > 2.0:
            return False
        for kind, center, size in self.obstacles:
            delta = np.abs(np.asarray(point) - center)
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                if np.all(delta <= size[:2] + self.clearance):
                    return False
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                if np.linalg.norm(delta) <= size[0] + self.clearance:
                    return False
            else:
                raise ValueError("Geometrie non prise en charge par le planificateur")
        return True

    def segment_free(self, a, b):
        count = max(2, math.ceil(math.dist(a, b) / 0.04) + 1)
        return all(self.free(p) for p in np.linspace(a, b, count))

    def plan(self, start, goal):
        def cell(point):
            return tuple(round(v / self.spacing) for v in point)

        def point(node):
            return tuple(v * self.spacing for v in node)

        source, destination = cell(start), cell(goal)
        if not self.segment_free(start, point(source)) or not self.segment_free(point(destination), goal):
            raise ValueError("Depart ou arrivee non accessible sur la grille")
        queue = [(math.dist(source, destination), 0.0, source)]
        costs, parents = {source: 0.0}, {}
        found = False
        while queue:
            _, cost, node = heapq.heappop(queue)
            if cost > costs[node]:
                continue
            if node == destination:
                found = True
                break
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nxt = (node[0] + dx, node[1] + dy)
                new_cost = cost + math.hypot(dx, dy)
                if new_cost >= costs.get(nxt, float("inf")):
                    continue
                if not self.segment_free(point(node), point(nxt)):
                    continue
                costs[nxt], parents[nxt] = new_cost, node
                heapq.heappush(queue, (new_cost + math.dist(nxt, destination), new_cost, nxt))
        if not found:
            raise ValueError("Aucun chemin libre vers l'arrivee")
        nodes = [destination]
        while nodes[-1] != source:
            nodes.append(parents[nodes[-1]])
        raw = [tuple(start)] + [point(n) for n in reversed(nodes)] + [tuple(goal)]
        # Greedy line-of-sight simplification, still checked against inflated obstacles.
        path, index = [raw[0]], 0
        while index < len(raw) - 1:
            nxt = len(raw) - 1
            while nxt > index + 1 and not self.segment_free(raw[index], raw[nxt]):
                nxt -= 1
            path.append(raw[nxt])
            index = nxt
        return path

    def decide(self, sim):
        x, y, yaw = sim.pose
        while self.target < len(self.path) - 1 and math.dist((x, y), self.path[self.target]) < 0.20:
            self.target += 1
        tx, ty = self.path[self.target]
        error = math.atan2(ty - y, tx - x) - yaw
        error = math.atan2(math.sin(error), math.cos(error))
        return "LEFT" if error > 0.15 else "RIGHT" if error < -0.15 else "STRAIGHT"
