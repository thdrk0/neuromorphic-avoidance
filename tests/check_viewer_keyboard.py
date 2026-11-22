"""Exercise real GLFW key events on an isolated Xvfb display (Linux only)."""

import ctypes as c
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    azerty = "--azerty" in sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Disable both pathname and abstract Unix sockets. Otherwise Xvfb can
        # shadow WSLg's :0 through an abstract socket and hide real GUI apps.
        with subprocess.Popen(["Xvfb", "-displayfd", "1", "-screen", "0", "1024x768x24",
                               "-nolisten", "local", "-nolisten", "unix",
                               "-listen", "tcp", "-ac"],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as server:
            try:
                display = "127.0.0.1:" + server.stdout.readline().decode().strip()
                env = dict(os.environ, DISPLAY=display, MUJOCO_GL="glfw")
                x = c.CDLL("libX11.so.6")
                xt = c.CDLL("libXtst.so.6")
                x.XOpenDisplay.argtypes = [c.c_char_p]
                x.XOpenDisplay.restype = c.c_void_p
                d = x.XOpenDisplay(display.encode())
                assert d, "Cannot open Xvfb display"
                subprocess.run(["setxkbmap", "-display", display, "fr" if azerty else "us"], check=True)
                x.XDefaultRootWindow.argtypes = [c.c_void_p]
                x.XDefaultRootWindow.restype = c.c_ulong
                x.XQueryTree.argtypes = [c.c_void_p, c.c_ulong, c.POINTER(c.c_ulong), c.POINTER(c.c_ulong),
                                         c.POINTER(c.POINTER(c.c_ulong)), c.POINTER(c.c_uint)]
                x.XFetchName.argtypes = [c.c_void_p, c.c_ulong, c.POINTER(c.c_char_p)]
                x.XSetInputFocus.argtypes = [c.c_void_p, c.c_ulong, c.c_int, c.c_ulong]
                x.XStringToKeysym.argtypes = [c.c_char_p]
                x.XStringToKeysym.restype = c.c_ulong
                x.XKeysymToKeycode.argtypes = [c.c_void_p, c.c_ulong]
                x.XKeysymToKeycode.restype = c.c_uint
                x.XFlush.argtypes = [c.c_void_p]
                x.XFree.argtypes = [c.c_void_p]
                x.XCloseDisplay.argtypes = [c.c_void_p]
                xt.XTestFakeKeyEvent.argtypes = [c.c_void_p, c.c_uint, c.c_int, c.c_ulong]

                with (root / "stdout").open("w") as log:
                    proc = subprocess.Popen([sys.executable, "-u", "-m", "neuroavoid.manual",
                                             "--debug-keys", "--view", "onboard" if "--onboard" in sys.argv else "overview",
                                             "--output", str(root / "output")], env=env,
                                            stdout=log, stderr=subprocess.STDOUT)
                    try:
                        deadline = time.monotonic() + 30
                        window = None
                        while time.monotonic() < deadline and proc.poll() is None:
                            parent, ancestor, count = c.c_ulong(), c.c_ulong(), c.c_uint()
                            children = c.POINTER(c.c_ulong)()
                            x.XQueryTree(d, x.XDefaultRootWindow(d), c.byref(ancestor), c.byref(parent),
                                         c.byref(children), c.byref(count))
                            for i in range(count.value):
                                name = c.c_char_p()
                                if x.XFetchName(d, children[i], c.byref(name)) and name.value:
                                    title = name.value.decode(errors="replace")
                                    x.XFree(name)
                                    if title == "MuJoCo - Pilotage robot":
                                        window = children[i]
                            if children:
                                x.XFree(children)
                            if window:
                                break
                            time.sleep(0.1)
                        assert window, "Viewer window not found"
                        x.XSetInputFocus(d, window, 1, 0)

                        def press(key):
                            code = x.XKeysymToKeycode(d, x.XStringToKeysym(key.encode()))
                            xt.XTestFakeKeyEvent(d, code, 1, 0)
                            xt.XTestFakeKeyEvent(d, code, 0, 0)
                            x.XFlush(d)

                        time.sleep(1)
                        press("z" if azerty else "w")
                        # Wait for simulated movement, independent of software-rendering speed.
                        deadline = time.monotonic() + 90
                        while time.monotonic() < deadline:
                            lines = (root / "stdout").read_text().splitlines()
                            states = [json.loads(line[7:]) for line in lines if line.startswith("Etat : ")]
                            if any(state["pose"][0] > -1.7 for state in states):
                                break
                            if proc.poll() is not None:
                                raise AssertionError((root / "stdout").read_text())
                            time.sleep(0.1)
                        else:
                            raise AssertionError("No keyboard-driven movement: " + (root / "stdout").read_text())
                        press("q" if azerty else "a")
                        time.sleep(0.5)
                        press("d")
                        time.sleep(0.5)
                        press("s")
                        time.sleep(0.5)
                        press("c")
                        time.sleep(0.5)
                        press("x")
                        try:
                            proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            print((root / "stdout").read_text())
                            raise
                        output = (root / "stdout").read_text()
                        print(output)
                        assert proc.returncode == 0
                        report = json.loads((root / "output/run.json").read_text())
                        assert "Commande : STRAIGHT" in output
                        assert "Commande : LEFT" in output
                        assert "Commande : RIGHT" in output
                        assert "Commande : STOP" in output
                        assert report["command"] == "STOP"
                        assert report["viewer"]["frames"] > 0
                        assert report["viewer"]["window_visible"]
                        assert report["viewer"]["display_flags_unchanged"], report
                        assert report["pose_xy_yaw"][0] > -1.8, report
                        from PIL import Image
                        import numpy as np
                        pixels = np.asarray(Image.open(root / "output/viewer_snapshot.png"))
                        assert pixels.mean() > 5 and pixels.std() > 10, "Black or uniform viewer image"
                        Path("artifacts").mkdir(exist_ok=True)
                        Image.fromarray(pixels).save("artifacts/viewer_check.png")
                        print("OK : real viewer keyboard moves robot")
                    finally:
                        if proc.poll() is None:
                            proc.terminate()
                            proc.wait(timeout=10)
                        x.XCloseDisplay(d)
            finally:
                server.terminate()
                server.wait(timeout=10)


if __name__ == "__main__":
    main()
