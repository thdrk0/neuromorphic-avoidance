# Third-party software

`third_party/v2e` contains a subset of SensorsINI/v2e, MIT licensed, with its
original copyright and LICENSE preserved. See its PROVENANCE.md for the exact
commit and the two deferred optional-export imports. Event generation is unchanged.

MuJoCo, PyTorch, NumPy, Pillow, Numba, OpenCV, Matplotlib, ROS2 and other Python
dependencies remain distributed under their respective upstream licenses.
Their installed packages and compiled ROS2 files are not part of this repository.

`third_party/wheels` is an optional local installation cache. Wheel binaries are
excluded from Git; their PyPI sources and checksums are documented there.

The two small checkpoints in `models` were trained for this project on its
synthetic MuJoCo/v2e data. No pretrained third-party network or Super SloMo
checkpoint is redistributed.
