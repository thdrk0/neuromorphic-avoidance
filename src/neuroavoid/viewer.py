"""Small MuJoCo renderer with application-owned keyboard handling."""

import glfw
import mujoco
import numpy as np


class RobotViewer:
    def __init__(self, model, data, key_callback):
        self.model, self.data = model, data
        if not glfw.init():
            raise RuntimeError("Impossible d'initialiser GLFW")
        # The offscreen Renderer changes global window hints. Start from defaults.
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        self.window = glfw.create_window(1000, 700, "MuJoCo - Pilotage robot", None, None)
        if not self.window:
            raise RuntimeError("Impossible de creer la fenetre de pilotage")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.cam = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=1000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.frames = 0
        self.initial_flags = None
        self.flags_unchanged = True
        self.help_text = ("Z/W: avancer  Q/A: gauche  D: droite  S/espace: arret\n"
                          "R: recommencer  C: photo  X: quitter")

        def on_key(window, key, scancode, action, mods):
            if action == glfw.PRESS:
                key_callback(key)

        # No native Simulate UI: keyboard events have exactly one handler.
        glfw.set_key_callback(self.window, on_key)
        glfw.set_window_pos(self.window, 100, 100)
        glfw.show_window(self.window)
        glfw.restore_window(self.window)
        glfw.focus_window(self.window)
        glfw.poll_events()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        glfw.make_context_current(self.window)
        self.context.free()
        glfw.destroy_window(self.window)
        # The offscreen camera Renderer also uses GLFW; do not terminate it here.

    def is_running(self):
        glfw.poll_events()
        return not glfw.window_should_close(self.window)

    def sync(self):
        # Camera rendering uses another OpenGL context. Restore this one explicitly.
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width == 0 or height == 0:
            return
        mujoco.mjv_updateScene(self.model, self.data, self.option, None, self.cam,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        flags = (tuple(self.scene.flags), tuple(self.option.flags),
                 tuple(self.option.geomgroup), int(self.cam.type))
        if self.initial_flags is None:
            self.initial_flags = flags
        self.flags_unchanged &= flags == self.initial_flags
        rect = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)
        mujoco.mjr_render(rect, self.scene, self.context)
        mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100,
                           mujoco.mjtGridPos.mjGRID_TOPLEFT, rect,
                           self.help_text, "", self.context)
        glfw.swap_buffers(self.window)
        self.frames += 1

    def screenshot(self):
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        rect = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)
        mujoco.mjv_updateScene(self.model, self.data, self.option, None, self.cam,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(rect, self.scene, self.context)
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(pixels, None, rect, self.context)
        return np.flipud(pixels).copy()

    def diagnostics(self):
        return {"frames": self.frames, "display_flags_unchanged": bool(self.flags_unchanged),
                "window_visible": bool(glfw.get_window_attrib(self.window, glfw.VISIBLE))}
