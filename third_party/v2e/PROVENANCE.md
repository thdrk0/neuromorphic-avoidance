# v2e core

Source: https://github.com/SensorsINI/v2e
Commit: cac638444557a224ba6ee5c29260916f152c00bd
License: MIT, see LICENSE.

Only the Python modules required by EventEmulator are included. The upstream
CLI, Super SloMo models and example recordings are not included.

Local patch: in emulator.py, import h5py only when HDF5 output is requested,
and import AEDat4Output only when AEDAT4 output is requested. Neither exporter
is used by this project. Event generation and its numerical code are unchanged.

The integration uses CPU mode with threshold mismatch, leak and shot noise
disabled. Frames come directly from MuJoCo; no learned frame interpolation.
