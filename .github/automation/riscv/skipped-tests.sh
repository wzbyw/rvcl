#!/usr/bin/env bash
# *******************************************************************************
# RVCL — RISC-V Compute Library
# SPDX-License-Identifier: Apache-2.0
#
# Tests to skip under QEMU emulation, by build tier (RVCL_CI_TIER).
#
# Emulated wall clock runs 20-100x native, so the skip list is tier-aware
# from day one: corpus growth lands here, not in ad-hoc workflow edits.
#   baseline — the C-fallback numerics are covered by the fixed
#              differential sections plus the smoke corpus; the full ci
#              corpus under emulation is reserved for the vector tier,
#              where it exercises the intrinsics tail paths (panel=8
#              two-strip at vlen=128, single-strip at vlen=256).
#   vector   — nothing skipped today. Performance-class targets (bench
#              P mode) get excluded here when they land in ctest.
# *******************************************************************************

set -eo pipefail

SKIPPED_TESTS=""
if [[ "${RVCL_CI_TIER}" == "baseline" ]]; then
    SKIPPED_TESTS="test_matmul_modeC_ci"
fi

echo "${SKIPPED_TESTS}"
