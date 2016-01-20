#!/bin/bash

BUILD_DIR="dir"

echo "=> Configure Eigen"
cd ./benchmark/eigen/
echo " -> Init submodule"
git submodule init
echo " -> Update submodule"
git submodule update

[ -d "${BUILD_DIR}" ] || ${BUILD_DIR}
cd ${BUILD_DIR}
cmake -DCMAKE_INSTALL_PREFIX="./" ..
make install