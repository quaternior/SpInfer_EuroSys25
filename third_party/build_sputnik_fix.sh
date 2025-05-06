#!/bin/bash

set -e

echo "[INFO] Starting glog + sputnik build (clean install-based)..."

GLOG_SRC_DIR="${SpInfer_HOME}/third_party/glog"
GLOG_INSTALL_DIR="${GLOG_SRC_DIR}/install"

SPUTNIK_SRC_DIR="${SpInfer_HOME}/third_party/sputnik"
SPUTNIK_BUILD_DIR="${SPUTNIK_SRC_DIR}/build"

echo "[INFO] Cleaning old glog build..."
rm -rf "${GLOG_SRC_DIR}/build" "${GLOG_INSTALL_DIR}"
mkdir -p "${GLOG_SRC_DIR}/build" "${GLOG_INSTALL_DIR}"

echo "[INFO] Building and installing glog..."
cd "${GLOG_SRC_DIR}/build"
cmake .. -DCMAKE_INSTALL_PREFIX="${GLOG_INSTALL_DIR}" -DBUILD_TESTING=OFF
make -j$(nproc)
make install

echo "[INFO] Setting environment variables..."
export GLOG_PATH="${GLOG_INSTALL_DIR}"
export LD_LIBRARY_PATH="${GLOG_PATH}/lib:${LD_LIBRARY_PATH}"
export CPLUS_INCLUDE_PATH="${GLOG_PATH}/include:${CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="${GLOG_PATH}/lib:${LIBRARY_PATH}"

echo "[INFO] Cleaning old sputnik build..."
rm -rf "${SPUTNIK_BUILD_DIR}"
mkdir -p "${SPUTNIK_BUILD_DIR}"
cd "${SPUTNIK_BUILD_DIR}"

echo "[INFO] Building sputnik with installed glog..."
cmake .. \
  -DGLOG_INCLUDE_DIR="${GLOG_PATH}/include" \
  -DGLOG_LIBRARY="${GLOG_PATH}/lib/libglog.so" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-DGLOG_EXPORT" \
  -DCMAKE_CXX_STANDARD=17 \
  -DBUILD_TEST=OFF \
  -DBUILD_BENCHMARK=OFF \
  -DCUDA_ARCHS="89;86;80"

make -j$(nproc)

echo "[SUCCESS] Build finished without errors."
