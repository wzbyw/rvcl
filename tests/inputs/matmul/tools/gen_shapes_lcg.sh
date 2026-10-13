#!/bin/sh
# RVCL — deterministic LCG shape generator for matmul corpora (zero deps).
#
# Usage: gen_shapes_lcg.sh <seed> <count> <maxdim>
#   Emits <count> lines `MxNxK n"lcgNN"` (1 <= each dim <= maxdim) on stdout.
#
# Lehmer RNG s' = (s * 48271) mod (2^31 - 1): the multiply stays below
# 2^53, so awk doubles are exact and every awk produces identical output.
# Record the seed/bounds in the generated corpus tail comment so the set
# can be rebuilt byte-for-byte (see test_matmul_ci for an example).
set -eu

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <seed> <count> <maxdim>" >&2
    exit 2
fi

awk -v seed="$1" -v count="$2" -v maxdim="$3" 'BEGIN {
    m = 2147483647
    a = 48271
    s = (seed + 0) % m
    if (s <= 0) s += m - 1
    maxdim += 0
    for (i = 1; i <= count; i++) {
        s = (s * a) % m; mi = 1 + s % maxdim
        s = (s * a) % m; ni = 1 + s % maxdim
        s = (s * a) % m; ki = 1 + s % maxdim
        printf "%dx%dx%d n\"lcg%02d\"\n", mi, ni, ki, i
    }
}'
