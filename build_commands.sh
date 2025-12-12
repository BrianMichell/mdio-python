#!/bin/bash
#
# source the .system venv first
# cd to mdio-cpp root
# run these commands
# cd back to mdio-python root

cmake -S . -B build-venv \
  -DPython3_ROOT_DIR="/home/ubuntu/source/mdio-python/.system" \
  -DPython3_EXECUTABLE="/home/ubuntu/source/mdio-python/.system/bin/python" \
  -DPython3_FIND_STRATEGY=LOCATION \
  -DMDIO_BUILD_PYBIND=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-venv --target mdio_cpp -j16

export PYTHONPATH=/home/ubuntu/source/mdio-cpp/build-venv/python:$PYTHONPATH
