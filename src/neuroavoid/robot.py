"""Differential-drive robot and physical contact reporting."""

from importlib.resources import files
import math

import mujoco
import numpy as np

from neuroavoid.config import (CAMERA_FPS, FORWARD_SPEED, PHYSICS_TIMESTEP,
                               TURN_SPEED, WHEEL_RADIUS, WHEEL_TRACK)

COMMANDS = {"STOP": (0.0, 0.0), "STRAIGHT": (FORWARD_SPEED, 0.0),
            "LEFT": (0.18, TURN_SPEED), "RIGHT": (0.18, -TURN_SPEED)}


class RobotSimulation:
    def __init__(self, scene_xml=None):
        self.model = (mujoco.MjModel.from_xml_string(scene_xml) if scene_xml is not None else
                      mujoco.MjModel.from_xml_path(str(files("neuroavoid") / "assets/arena.xml")))
        self.model.opt.timestep = PHYSICS_TIMESTEP
        self.data = mujoco.MjData(self.model)
        self.camera_stride = round(1 / CAMERA_FPS / PHYSICS_TIMESTEP)
        self.robot_body = self.model.body("robot").id
        self.robot_geoms = set()
        for geom in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[geom])
            while body:
                if body == self.robot_body:
                    self.robot_geoms.add(geom)
                    break
                body = int(self.model.body_parentid[body])
        self.floor = self.model.geom("floor").id
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.collision_steps = 0
        self.collided = False
        self.contacts = set()
        self.command = "STOP"

    def set_command(self, command, speed_factor=1.0):
        if not math.isfinite(speed_factor) or not 0 < speed_factor <= 1.9:
            raise ValueError("speed_factor doit etre dans ]0,1.9] pour eviter la saturation des moteurs")
        v, omega = COMMANDS[command]
        v, omega = v * speed_factor, omega * speed_factor
        self.command = command
        self.data.ctrl[:] = [(v - omega * WHEEL_TRACK / 2) / WHEEL_RADIUS,
                             (v + omega * WHEEL_TRACK / 2) / WHEEL_RADIUS]

    def step(self):
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        contacts = set()
        for contact in self.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if (a in self.robot_geoms) != (b in self.robot_geoms):
                other = b if a in self.robot_geoms else a
                if other != self.floor and contact.dist <= 0:
                    contacts.add(self.model.geom(other).name)
        self.contacts = contacts
        if contacts:
            self.collided = True
            self.collision_steps += 1
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("Etat physique non fini")
        return self.steps % self.camera_stride == 0

    @property
    def pose(self):
        w, x, y, z = self.data.qpos[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return [float(self.data.qpos[0]), float(self.data.qpos[1]), yaw]
