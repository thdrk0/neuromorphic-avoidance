#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-73}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
command="${1:-demo}"
if [ "$#" -gt 0 ]; then shift; fi
if [ "$command" = build ]; then
  cd ros_ws
  colcon build --packages-select neuroavoid_interfaces --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
  exit
fi
if [ ! -f ros_ws/install/setup.bash ]; then
  echo 'Compiler les interfaces : bash scripts/run_ros.sh build' >&2
  exit 1
fi
source ros_ws/install/setup.bash
case "$command" in
  demo) exec .venv/bin/python -u -m neuroavoid.ros_demo "$@" ;;
  policy|sim) exec .venv/bin/python -u -m neuroavoid.ros_nodes "$command" "$@" ;;
  cli) exec ros2 "$@" ;;
  check-failure) exec .venv/bin/python -u tests/check_ros_failure.py ;;
  *) echo 'Usage : run_ros.sh build|demo|policy|sim|cli' >&2; exit 1 ;;
esac
