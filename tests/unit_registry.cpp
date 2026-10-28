/*******************************************************************************
 * Copyright 2026 ZTE Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * unit_registry.cpp — registry unit tests (RFC-0002 FR1 / AC1)
 *
 * Covers: full-field registration, lazy init via first computation, init
 * idempotence, duplicate/capacity/malformed rejection, iteration and
 * factory contracts, unregister semantics incl. the per-tuple fallback
 * invariant, builtin registration order, and concurrent
 * register/unregister/query safety (meaningful under the ASan build).
 *
 * Custom kernels use datatype tuples no builtin occupies (INT8/INT4) and
 * are unregistered again at the end of each case, followed by a dispatch
 * cache clear, so cases stay independent.
 ******************************************************************************/

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "golden.hpp"
#include "rvcl/rvcl.h"

static int g_failures;

#define CHECK(cond, msg) \
    do { \
        if (cond) { \
            printf("  ok: %s\n", msg); \
        } else { \
            printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
            g_failures++; \
        } \
    } while (0)

/* ------------------------------------------------------- minimal test kernel
 *
 * A do-nothing kernel: none of these cases execute it; it only has to
 * satisfy the factory contract. */

static rvcl_status_t test_validate(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)desc;
    (void)cap;
    return RVCL_SUCCESS;
}

static rvcl_status_t test_configure(rvcl_kernel_t *k,
        const rvcl_op_desc_t *desc, rvcl_workspace_req_t *ws_req, int *num_ws) {
    (void)k;
    (void)desc;
    if (ws_req) *ws_req = rvcl_workspace_req_t {};
    if (num_ws) *num_ws = 0;
    return RVCL_SUCCESS;
}

static rvcl_status_t test_run(rvcl_kernel_t *k, const rvcl_tensor_pack_t *pack,
        const rvcl_work_range_t *range) {
    (void)k;
    (void)pack;
    (void)range;
    return RVCL_SUCCESS;
}

static void test_destroy(rvcl_kernel_t *k) {
    free(k);
}

static const rvcl_kernel_vtable_t g_test_vtable = {
        .validate = test_validate,
        .configure = test_configure,
        .pack_a = NULL,
        .pack_b = NULL,
        .run = test_run,
        .destroy = test_destroy,
};

static rvcl_status_t test_create(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    *out = (rvcl_kernel_t *)calloc(1, sizeof(rvcl_kernel_t));
    if (!*out) return RVCL_ERROR_OUT_OF_MEMORY;
    (*out)->vt = &g_test_vtable;
    (*out)->desc = *desc;
    return RVCL_SUCCESS;
}

/* ------------------------------------------------------------- helpers */

static rvcl_kernel_desc_t make_desc(const char *name) {
    rvcl_kernel_desc_t d;
    memset(&d, 0, sizeof(d));
    /* precision-bounded copy: GCC cannot prove arbitrary callers fit the
     * fixed name buffer, and -Wformat-truncation is fatal under Werror */
    snprintf(d.name, RVCL_KERNEL_NAME_MAX, "%.*s", RVCL_KERNEL_NAME_MAX - 1,
            name);
    d.operation = RVCL_OP_MATMUL;
    d.dtype_a = RVCL_DTYPE_INT8;
    d.dtype_b = RVCL_DTYPE_INT8;
    d.dtype_acc = RVCL_DTYPE_INT32;
    d.dtype_c = RVCL_DTYPE_INT32;
    d.quant_support = RVCL_QUANT_NONE;
    d.num_required_features = 0;
    d.alignment_a = d.alignment_b = d.alignment_c = 1;
    d.vlen_tier = RVCL_VLEN_TIER_ANY;
    d.cost_fn = rvcl_kernel_cost_zero;
    return d;
}

static void unregister_quiet(const char *name) {
    rvcl_unregister_kernel(name);
}

static int find_name_index(const char *name) {
    const int n = rvcl_registry_count();
    for (int i = 0; i < n; i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        if (kd && strcmp(kd->name, name) == 0) return i;
    }
    return -1;
}

static int vector_build(void) {
    return rvcl_registry_find_factory("rvcl_matmul_f32_rvv") != NULL;
}

/* TC-AC1-02: first computation triggers initialization (must run before
 * any explicit rvcl_init in this process). */
static void tc02_first_compute_lazy_init(void) {
    printf("TC-AC1-02 first computation triggers init\n");
    enum { M = 8, N = 8, K = 16 };
    float a[M * K], b[K * N], bias[N], c[M * N], ref[M * N];
    for (int i = 0; i < M * K; i++)
        a[i] = (float)((i * 13) % 29) / 29.0f - 0.5f;
    for (int i = 0; i < K * N; i++)
        b[i] = (float)((i * 7) % 31) / 31.0f - 0.5f;
    for (int j = 0; j < N; j++)
        bias[j] = 0.25f;

    rvcl_matmul_desc_t d = {
            .m = M,
            .n = N,
            .k = K,
            .dtype_a = RVCL_DTYPE_FP32,
            .dtype_b = RVCL_DTYPE_FP32,
            .dtype_acc = RVCL_DTYPE_AUTO,
            .dtype_c = RVCL_DTYPE_FP32,
            .dtype_bias = RVCL_DTYPE_FP32,
            .quant = {},
    };
    CHECK(rvcl_matmul(&d, a, b, bias, c) == RVCL_SUCCESS,
            "first computation succeeds without explicit init");
    golden_f32(a, b, bias, ref, M, N, K);
    CHECK(memcmp(c, ref, sizeof(c)) == 0, "lazy-init path matches golden");

    CHECK(rvcl_registry_count() >= 1, "registry populated after lazy init");
    CHECK(rvcl_registry_find_factory("rvcl_matmul_f32_reference") != NULL,
            "f32 fallback entry present");
}

/* TC-AC1-01: legal descriptor registers with every field intact. */
static void tc01_register_full_fields(void) {
    printf("TC-AC1-01 full-field registration\n");
    const int c0 = rvcl_registry_count();
    rvcl_kernel_desc_t d = make_desc("t01_full");
    d.required_features[0] = RVCL_FEATURE_INT8;
    d.required_features[1] = RVCL_FEATURE_RV64;
    d.num_required_features = 2;
    d.min_m = 2;
    d.max_m = 64;
    d.min_n = 1;
    d.max_n = 0; /* unbounded */
    d.min_k = 4;
    d.max_k = 512;
    d.preferred_m = 8;
    d.preferred_n = 8;
    d.preferred_k = 8;
    d.alignment_a = 2;
    d.alignment_b = 4;
    d.alignment_c = 2;
    d.vlen_tier = RVCL_VLEN_TIER_256;
    d.cost_fn = rvcl_kernel_cost_zero;

    CHECK(rvcl_register_kernel(&d, test_create) == RVCL_SUCCESS,
            "valid descriptor registers");
    CHECK(rvcl_registry_count() == c0 + 1, "count incremented by one");

    const int idx = find_name_index("t01_full");
    CHECK(idx >= 0, "entry iterable by name");
    if (idx >= 0) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(idx);
        int same = kd && kd->operation == d.operation
                && kd->dtype_a == d.dtype_a && kd->dtype_b == d.dtype_b
                && kd->dtype_acc == d.dtype_acc && kd->dtype_c == d.dtype_c
                && kd->quant_support == d.quant_support
                && kd->num_required_features == d.num_required_features
                && kd->required_features[0] == d.required_features[0]
                && kd->required_features[1] == d.required_features[1]
                && kd->min_m == d.min_m && kd->max_m == d.max_m
                && kd->min_n == d.min_n && kd->max_n == d.max_n
                && kd->min_k == d.min_k && kd->max_k == d.max_k
                && kd->preferred_m == d.preferred_m
                && kd->preferred_n == d.preferred_n
                && kd->preferred_k == d.preferred_k
                && kd->alignment_a == d.alignment_a
                && kd->alignment_b == d.alignment_b
                && kd->alignment_c == d.alignment_c
                && kd->vlen_tier == d.vlen_tier
                && kd->cost_fn == rvcl_kernel_cost_zero;
        CHECK(same, "every registered field reads back identically");
    }
    unregister_quiet("t01_full");
    rvcl_dispatch_cache_clear();
    CHECK(rvcl_registry_count() == c0, "cleanup restores count");
}

/* TC-AC1-03: init idempotence; entries for variants that did not compile
 * into the build are absent. */
static void tc03_init_idempotent(void) {
    printf("TC-AC1-03 init idempotence and build guards\n");
    char before[64][RVCL_KERNEL_NAME_MAX];
    const int n0 = rvcl_registry_count();
    for (int i = 0; i < n0 && i < RVCL_REGISTRY_MAX_KERNELS; i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        snprintf(before[i], RVCL_KERNEL_NAME_MAX, "%s", kd ? kd->name : "");
    }

    CHECK(rvcl_init() == RVCL_SUCCESS, "repeated init succeeds");
    CHECK(rvcl_init() == RVCL_SUCCESS, "third init succeeds");
    CHECK(rvcl_registry_count() == n0, "count unchanged by re-init");

    int same = 1;
    for (int i = 0; i < n0; i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        if (!kd || strcmp(kd->name, before[i]) != 0) same = 0;
    }
    CHECK(same, "name sequence unchanged by re-init");

    if (!vector_build()) {
        int leaked = 0;
        for (int i = 0; i < rvcl_registry_count(); i++) {
            const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
            if (!kd) continue;
            if (kd->vlen_tier != RVCL_VLEN_TIER_ANY) leaked = 1;
            for (int f = 0; f < kd->num_required_features; f++)
                if (kd->required_features[f] == RVCL_FEATURE_RVV) leaked = 1;
        }
        CHECK(leaked == 0, "non-vector build has no RVV/tier entries");
    } else {
        printf("  ok: (vector build, tier entries expected)\n");
    }
}

/* TC-AC1-04: duplicate names rejected atomically. */
static void tc04_duplicate_rejected(void) {
    printf("TC-AC1-04 duplicate registration rejected\n");
    const int c0 = rvcl_registry_count();

    rvcl_kernel_desc_t a = make_desc("t04_dup");
    CHECK(rvcl_register_kernel(&a, test_create) == RVCL_SUCCESS,
            "first registration succeeds");

    rvcl_kernel_desc_t b = make_desc("t04_dup");
    b.max_k = 128; /* different content, same name */
    CHECK(rvcl_register_kernel(&b, test_create)
                    == RVCL_ERROR_ALREADY_REGISTERED,
            "duplicate name rejected");
    CHECK(rvcl_registry_count() == c0 + 1, "count unchanged by rejection");

    /* builtin names are reserved the same way */
    rvcl_kernel_desc_t c = make_desc("rvcl_matmul_f32_reference");
    c.dtype_a = RVCL_DTYPE_FP32;
    c.dtype_b = RVCL_DTYPE_FP32;
    c.dtype_acc = RVCL_DTYPE_FP32;
    c.dtype_c = RVCL_DTYPE_FP32;
    CHECK(rvcl_register_kernel(&c, test_create)
                    == RVCL_ERROR_ALREADY_REGISTERED,
            "builtin name reserved");

    const int idx = find_name_index("t04_dup");
    if (idx >= 0)
        CHECK(rvcl_registry_get(idx)->max_k == 0,
                "original fields intact after rejected duplicate");
    unregister_quiet("t04_dup");
    rvcl_dispatch_cache_clear();
}

/* TC-AC1-05: capacity limit rejects atomically. */
static void tc05_capacity(void) {
    printf("TC-AC1-05 capacity overflow rejected\n");
    const int c0 = rvcl_registry_count();
    char names[RVCL_REGISTRY_MAX_KERNELS][RVCL_KERNEL_NAME_MAX];
    int added = 0;

    for (int i = 0; c0 + added < RVCL_REGISTRY_MAX_KERNELS; i++) {
        snprintf(names[added], RVCL_KERNEL_NAME_MAX, "t05_cap_%d", i);
        rvcl_kernel_desc_t d = make_desc(names[added]);
        if (rvcl_register_kernel(&d, test_create) != RVCL_SUCCESS) break;
        added++;
    }
    CHECK(rvcl_registry_count() == RVCL_REGISTRY_MAX_KERNELS,
            "registry fills to capacity");

    rvcl_kernel_desc_t d = make_desc("t05_over");
    CHECK(rvcl_register_kernel(&d, test_create) == RVCL_ERROR_OUT_OF_MEMORY,
            "entry beyond capacity rejected");
    CHECK(rvcl_registry_count() == RVCL_REGISTRY_MAX_KERNELS,
            "count unchanged by overflow");

    int distinct = 1;
    for (int i = 0; i < RVCL_REGISTRY_MAX_KERNELS && distinct; i++)
        for (int j = i + 1; j < RVCL_REGISTRY_MAX_KERNELS; j++) {
            const rvcl_kernel_desc_t *a = rvcl_registry_get(i);
            const rvcl_kernel_desc_t *b = rvcl_registry_get(j);
            if (a && b && strcmp(a->name, b->name) == 0) distinct = 0;
        }
    CHECK(distinct, "all capacity entries have distinct names");

    for (int i = 0; i < added; i++)
        unregister_quiet(names[i]);
    rvcl_dispatch_cache_clear();
    CHECK(rvcl_registry_count() == c0, "cleanup restores count");
}

/* TC-AC1-06: malformed descriptors rejected with no side effects. */
static void tc06_invalid_descriptors(void) {
    printf("TC-AC1-06 malformed descriptors rejected\n");
    const int c0 = rvcl_registry_count();
    int rejects = 0, total = 0;

    struct {
        const char *label;
        rvcl_kernel_desc_t d;
    } cases[32];
    int nc = 0;

    rvcl_kernel_desc_t base = make_desc("t06_invalid");

#define ADD_CASE(label_str, mut) \
    do { \
        rvcl_kernel_desc_t d = base; \
        mut; \
        cases[nc].label = label_str; \
        cases[nc].d = d; \
        nc++; \
    } while (0)

    ADD_CASE("empty name", d.name[0] = '\0');
    ADD_CASE("overlong name", memset(d.name, 'x', RVCL_KERNEL_NAME_MAX));
    ADD_CASE("operation out of range", d.operation = RVCL_OP_COUNT);
    ADD_CASE("operation negative", d.operation = (rvcl_operation_t)-1);
    ADD_CASE("dtype_a AUTO", d.dtype_a = RVCL_DTYPE_AUTO);
    ADD_CASE("dtype_b AUTO", d.dtype_b = RVCL_DTYPE_AUTO);
    ADD_CASE("dtype_acc AUTO", d.dtype_acc = RVCL_DTYPE_AUTO);
    ADD_CASE("dtype_c AUTO", d.dtype_c = RVCL_DTYPE_AUTO);
    ADD_CASE("dtype out of range", d.dtype_a = (rvcl_dtype_t)99);
    ADD_CASE("quant out of range", d.quant_support = (rvcl_quant_support_t)99);
    ADD_CASE("too many features",
            d.num_required_features = RVCL_MAX_FEATURES + 1);
    ADD_CASE("feature NONE listed",
            (d.required_features[0] = RVCL_FEATURE_NONE,
                    d.num_required_features = 1));
    ADD_CASE("negative min_m", d.min_m = -1);
    ADD_CASE("min_m above max_m", (d.min_m = 8, d.max_m = 4));
    ADD_CASE("min_n above max_n", (d.min_n = 8, d.max_n = 4));
    ADD_CASE("min_k above max_k", (d.min_k = 8, d.max_k = 4));
    ADD_CASE("tier 64", d.vlen_tier = (rvcl_vlen_tier_t)64);
    ADD_CASE("tier 100", d.vlen_tier = (rvcl_vlen_tier_t)100);
    ADD_CASE("tier 3", d.vlen_tier = (rvcl_vlen_tier_t)3);
    ADD_CASE("tier negative", d.vlen_tier = (rvcl_vlen_tier_t)-1);
    ADD_CASE("NULL cost callback", d.cost_fn = NULL);

    for (int i = 0; i < nc; i++) {
        total++;
        if (rvcl_register_kernel(&cases[i].d, test_create)
                == RVCL_ERROR_INVALID_ARGUMENT)
            rejects++;
        else
            printf("  FAIL: case accepted: %s\n", cases[i].label);
    }
    CHECK(total == 21 && rejects == 21, "all malformed descriptors rejected");
    CHECK(rvcl_register_kernel(NULL, test_create)
                    == RVCL_ERROR_INVALID_ARGUMENT,
            "NULL descriptor rejected");
    rvcl_kernel_desc_t d = make_desc("t06_null_factory");
    CHECK(rvcl_register_kernel(&d, NULL) == RVCL_ERROR_INVALID_ARGUMENT,
            "NULL factory rejected");
    CHECK(rvcl_registry_count() == c0, "registry unchanged by all rejections");
}

/* TC-AC1-07: count / iteration / factory lookup contract. */
static void tc07_count_iterate_factory(void) {
    printf("TC-AC1-07 iteration and factory contract\n");
    rvcl_kernel_desc_t d = make_desc("t07_probe");
    rvcl_register_kernel(&d, test_create);

    const int n = rvcl_registry_count();
    CHECK(n >= 1, "count positive");
    CHECK(rvcl_registry_get(n) == NULL, "index == count returns NULL");
    CHECK(rvcl_registry_get(n + 1) == NULL, "index past count returns NULL");
    CHECK(rvcl_registry_get(-1) == NULL, "negative index returns NULL");

    int all_present = 1;
    for (int i = 0; i < n; i++)
        if (!rvcl_registry_get(i)) all_present = 0;
    CHECK(all_present, "every in-range index returns an entry");

    CHECK(rvcl_registry_find_factory("t07_probe") != NULL,
            "factory resolvable for registered name");
    CHECK(rvcl_registry_find_factory("t07_never_registered") == NULL,
            "unknown name yields NULL factory");
    unregister_quiet("t07_probe");
    rvcl_dispatch_cache_clear();
}

/* TC-AC1-08 / TC-AC1-09: unregister success and not-found paths. */
static void tc08_09_unregister(void) {
    printf("TC-AC1-08/09 unregister semantics\n");
    const int c0 = rvcl_registry_count();
    rvcl_kernel_desc_t d = make_desc("t08_x");
    CHECK(rvcl_register_kernel(&d, test_create) == RVCL_SUCCESS,
            "custom kernel registered");

    CHECK(rvcl_unregister_kernel("t08_x") == RVCL_SUCCESS,
            "unregister by name succeeds");
    CHECK(rvcl_registry_count() == c0, "count restored");
    CHECK(find_name_index("t08_x") < 0, "unregistered entry not iterable");
    CHECK(rvcl_registry_find_factory("t08_x") == NULL,
            "unregistered entry has no factory");

    CHECK(rvcl_unregister_kernel("no_such_kernel") == RVCL_ERROR_NOT_FOUND,
            "unregistering unknown name -> NOT_FOUND");
    CHECK(rvcl_unregister_kernel("t08_x") == RVCL_ERROR_NOT_FOUND,
            "double unregister -> NOT_FOUND");
    CHECK(rvcl_registry_count() == c0, "failed unregisters change nothing");
    rvcl_dispatch_cache_clear();
}

/* TC-AC1-10: unregistering cannot break the per-tuple fallback
 * invariant. Uses an INT4 tuple no builtin occupies. */
static void tc10_fallback_invariant(void) {
    printf("TC-AC1-10 fallback invariant on unregister\n");
    const int c0 = rvcl_registry_count();

    rvcl_kernel_desc_t bounded = make_desc("t10_b1");
    bounded.dtype_a = RVCL_DTYPE_INT4;
    bounded.dtype_b = RVCL_DTYPE_INT4;
    bounded.max_m = 64; /* bounded */
    rvcl_kernel_desc_t unbnd = make_desc("t10_u1");
    unbnd.dtype_a = RVCL_DTYPE_INT4;
    unbnd.dtype_b = RVCL_DTYPE_INT4;

    CHECK(rvcl_register_kernel(&bounded, test_create) == RVCL_SUCCESS
                    && rvcl_register_kernel(&unbnd, test_create)
                            == RVCL_SUCCESS,
            "bounded + unbounded pair registered");

    CHECK(rvcl_unregister_kernel("t10_u1") == RVCL_ERROR_REGISTRY_INVARIANT,
            "removing the only unbounded entry rejected");
    CHECK(rvcl_registry_find_factory("t10_u1") != NULL,
            "rejected unregister leaves the entry in place");

    rvcl_kernel_desc_t unbnd2 = unbnd;
    snprintf(unbnd2.name, RVCL_KERNEL_NAME_MAX, "%s", "t10_u2");
    CHECK(rvcl_register_kernel(&unbnd2, test_create) == RVCL_SUCCESS,
            "second unbounded entry registered");
    CHECK(rvcl_unregister_kernel("t10_u1") == RVCL_SUCCESS,
            "unregister allowed when another unbounded remains");

    CHECK(rvcl_unregister_kernel("t10_b1") == RVCL_SUCCESS,
            "bounded entry removable");
    CHECK(rvcl_unregister_kernel("t10_u2") == RVCL_SUCCESS,
            "last entry of an emptying tuple removable");

    /* builtin f32 tuple: reject removing a unique unbounded fallback,
     * skip when other unbounded entries keep the tuple covered. */
    int f32_unbounded = 0, reference_unbounded = 0;
    for (int i = 0; i < rvcl_registry_count(); i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        if (!kd || kd->dtype_a != RVCL_DTYPE_FP32) continue;
        if (kd->max_m == 0 && kd->max_n == 0 && kd->max_k == 0) {
            f32_unbounded++;
            if (strcmp(kd->name, "rvcl_matmul_f32_reference") == 0)
                reference_unbounded = 1;
        }
    }
    if (reference_unbounded && f32_unbounded == 1) {
        CHECK(rvcl_unregister_kernel("rvcl_matmul_f32_reference")
                        == RVCL_ERROR_REGISTRY_INVARIANT,
                "builtin unique fallback protected");
        CHECK(rvcl_registry_find_factory("rvcl_matmul_f32_reference") != NULL,
                "builtin fallback still present");
    } else {
        printf("  ok: (f32 tuple has other unbounded entries, skip)\n");
    }
    CHECK(rvcl_registry_count() == c0, "table restored");
    rvcl_dispatch_cache_clear();
}

/* TC-AC1-11: every live datatype tuple keeps an unbounded entry. */
static void tc11_table_invariant(void) {
    printf("TC-AC1-11 table-wide fallback invariant\n");
    const int n = rvcl_registry_count();
    int ok = 1, tuples = 0;
    for (int i = 0; i < n; i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        if (!kd) continue;
        tuples++;
        int covered = 0;
        for (int j = 0; j < n; j++) {
            const rvcl_kernel_desc_t *o = rvcl_registry_get(j);
            if (!o || o->operation != kd->operation || o->dtype_a != kd->dtype_a
                    || o->dtype_b != kd->dtype_b
                    || o->dtype_acc != kd->dtype_acc
                    || o->dtype_c != kd->dtype_c)
                continue;
            if (o->max_m == 0 && o->max_n == 0 && o->max_k == 0) covered = 1;
        }
        if (!covered) ok = 0;
    }
    CHECK(ok && tuples == n, "every live tuple has an unbounded fallback");
}

/* TC-AC1-12: builtin order = dedicated (high tier -> low), then generic,
 * then fallback. */
static void tc12_builtin_order(void) {
    printf("TC-AC1-12 builtin registration order\n");
    const int n = rvcl_registry_count();
    int ok = 1;

    for (int pass = 0; pass < 2; pass++) {
        const rvcl_dtype_t key = pass == 0 ? RVCL_DTYPE_FP32 : RVCL_DTYPE_FP16;
        int prev_cls = -1; /* 0 dedicated, 1 generic, 2 fallback */
        int prev_tier = 1 << 30;
        int seen = 0;
        int group_ok = 1;
        for (int i = 0; i < n; i++) {
            const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
            if (!kd || kd->dtype_a != key || kd->dtype_b != key) continue;
            seen++;

            int cls;
            if (kd->vlen_tier != RVCL_VLEN_TIER_ANY) {
                cls = 0;
                /* tiers must strictly descend among consecutive
                 * dedicated entries */
                if (prev_cls == 0 && (int)kd->vlen_tier >= prev_tier)
                    group_ok = 0;
                prev_tier = (int)kd->vlen_tier;
            } else {
                int needs_rvv = 0;
                for (int f = 0; f < kd->num_required_features; f++)
                    if (kd->required_features[f] == RVCL_FEATURE_RVV)
                        needs_rvv = 1;
                cls = needs_rvv ? 1 : 2;
            }
            if (cls < prev_cls) group_ok = 0; /* never go backwards */
            prev_cls = cls;
        }
        if (pass == 0) CHECK(group_ok && seen >= 2, "f32 group ordered");
        if (pass == 1) CHECK(group_ok && seen >= 1, "f16 group ordered");
        if (!group_ok) ok = 0;
    }
    (void)ok; /* per-group CHECKs above carry the verdict */
}

/* TC-AC1-13: concurrent register / unregister / query. The interesting
 * race window is exercised under the ASan build. */

#define T13_THREADS_A 8
#define T13_PER_A 4
#define T13_THREADS_B 4
#define T13_THREADS_C 4
#define T13_QUERIES 2000

typedef struct {
    int tid;
    int reg_ok;
    int unreg_ok;
} t13_arg_t;

static void *t13_register(void *p) {
    t13_arg_t *arg = (t13_arg_t *)p;
    for (int i = 0; i < T13_PER_A; i++) {
        char name[RVCL_KERNEL_NAME_MAX];
        snprintf(name, sizeof(name), "t13_a_%d_%d", arg->tid, i);
        rvcl_kernel_desc_t d = make_desc(name);
        if (rvcl_register_kernel(&d, test_create) == RVCL_SUCCESS)
            arg->reg_ok++;
    }
    return NULL;
}

static void *t13_unregister(void *p) {
    t13_arg_t *arg = (t13_arg_t *)p;
    for (int i = 0; i < T13_PER_A / 2; i++) {
        char name[RVCL_KERNEL_NAME_MAX];
        snprintf(name, sizeof(name), "t13_a_%d_%d", arg->tid, i);
        if (rvcl_unregister_kernel(name) == RVCL_SUCCESS) arg->unreg_ok++;
    }
    return NULL;
}

static void *t13_query(void *p) {
    t13_arg_t *arg = (t13_arg_t *)p;
    (void)arg;
    for (int i = 0; i < T13_QUERIES; i++) {
        const int n = rvcl_registry_count();
        for (int j = 0; j < n; j += 7) {
            const rvcl_kernel_desc_t *kd = rvcl_registry_get(j);
            if (kd && kd->name[0] == '\0') return (void *)1; /* torn read */
        }
        rvcl_registry_find_factory("t13_a_0_0");
    }
    return NULL;
}

static void *t13_same_name_unregister(void *p) {
    t13_arg_t *arg = (t13_arg_t *)p;
    if (rvcl_unregister_kernel("t13_dup_race") == RVCL_SUCCESS) arg->unreg_ok++;
    return NULL;
}

static void tc13_concurrent(void) {
    printf("TC-AC1-13 concurrent register/unregister/query\n");
    const int c0 = rvcl_registry_count();

    /* same-name race target (counted explicitly in the arithmetic below;
     * its removal by the D-group winner is why c0 itself must not include
     * it) */
    rvcl_kernel_desc_t dup = make_desc("t13_dup_race");
    CHECK(rvcl_register_kernel(&dup, test_create) == RVCL_SUCCESS,
            "same-name race target registered");

    pthread_t ta[T13_THREADS_A], tb[T13_THREADS_B], tc[T13_THREADS_C], td[2];
    t13_arg_t aa[T13_THREADS_A], ab[T13_THREADS_B], ac[T13_THREADS_C], ad[2];

    for (int i = 0; i < T13_THREADS_A; i++) {
        aa[i] = t13_arg_t {.tid = i, .reg_ok = 0, .unreg_ok = 0};
        pthread_create(&ta[i], NULL, t13_register, &aa[i]);
    }
    for (int i = 0; i < T13_THREADS_B; i++) {
        ab[i] = t13_arg_t {.tid = i, .reg_ok = 0, .unreg_ok = 0};
        pthread_create(&tb[i], NULL, t13_unregister, &ab[i]);
    }
    for (int i = 0; i < T13_THREADS_C; i++) {
        ac[i] = t13_arg_t {.tid = i, .reg_ok = 0, .unreg_ok = 0};
        pthread_create(&tc[i], NULL, t13_query, &ac[i]);
    }
    /* two racers on the same name */
    for (int i = 0; i < 2; i++) {
        ad[i] = t13_arg_t {.tid = i, .reg_ok = 0, .unreg_ok = 0};
        pthread_create(&td[i], NULL, t13_same_name_unregister, &ad[i]);
    }

    int reg_ok = 0, unreg_ok = 0, torn = 0;
    for (int i = 0; i < T13_THREADS_A; i++) {
        pthread_join(ta[i], NULL);
        reg_ok += aa[i].reg_ok;
    }
    for (int i = 0; i < T13_THREADS_B; i++) {
        pthread_join(tb[i], NULL);
        unreg_ok += ab[i].unreg_ok;
    }
    for (int i = 0; i < T13_THREADS_C; i++) {
        void *ret = NULL;
        pthread_join(tc[i], &ret);
        if (ret) torn = 1;
    }
    int dup_ok = 0;
    for (int i = 0; i < 2; i++) {
        pthread_join(td[i], NULL);
        dup_ok += ad[i].unreg_ok;
    }

    CHECK(reg_ok == T13_THREADS_A * T13_PER_A, "all concurrent registers ok");
    CHECK(dup_ok == 1, "exactly one winner for the same-name race");
    CHECK(torn == 0, "concurrent queries saw no torn descriptors");
    CHECK(rvcl_registry_count() == c0 + 1 + reg_ok - unreg_ok - dup_ok,
            "final count matches successful operations");

    /* cleanup the survivors */
    for (int t = 0; t < T13_THREADS_A; t++)
        for (int i = 0; i < T13_PER_A; i++) {
            char name[RVCL_KERNEL_NAME_MAX];
            snprintf(name, sizeof(name), "t13_a_%d_%d", t, i);
            unregister_quiet(name);
        }
    rvcl_dispatch_cache_clear();
    CHECK(rvcl_registry_count() == c0, "cleanup restores count");
}

/* TC-AC1-14: builtin entries exist per build configuration and every
 * name resolves. */
static void tc14_builtin_entries(void) {
    printf("TC-AC1-14 builtin entries present and queryable\n");
    const int n = rvcl_registry_count();
    int all_resolvable = 1;
    for (int i = 0; i < n; i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        if (!kd || !rvcl_registry_find_factory(kd->name)) all_resolvable = 0;
    }
    CHECK(all_resolvable, "every registry name resolves to a factory");

    CHECK(rvcl_registry_find_factory("rvcl_matmul_f32_reference") != NULL,
            "f32 unbounded reference present");

    if (vector_build()) {
        static const int tiers[4] = {128, 256, 512, 1024};
        for (int t = 0; t < 4; t++) {
            char name[RVCL_KERNEL_NAME_MAX];
            snprintf(name, sizeof(name), "rvcl_matmul_f32_rvv_%d", tiers[t]);
            const int idx = find_name_index(name);
            CHECK(idx >= 0, "per-tier f32 rvv entry present");
            if (idx >= 0)
                CHECK(rvcl_registry_get(idx)->vlen_tier == tiers[t],
                        "tier entry declares its tier");
        }
    } else {
        CHECK(rvcl_registry_find_factory("rvcl_matmul_f32_rvv_c") != NULL,
                "portable f32 entry present on non-vector builds");
        CHECK(rvcl_registry_find_factory("rvcl_matmul_f16_rvv_c") != NULL,
                "portable f16 entry present on non-vector builds");
    }
}

int main(void) {
    tc02_first_compute_lazy_init(); /* must be first: fresh-process lazy init */
    CHECK(rvcl_init() == RVCL_SUCCESS, "explicit init after lazy path");
    tc01_register_full_fields();
    tc03_init_idempotent();
    tc04_duplicate_rejected();
    tc05_capacity();
    tc06_invalid_descriptors();
    tc07_count_iterate_factory();
    tc08_09_unregister();
    tc10_fallback_invariant();
    tc11_table_invariant();
    tc12_builtin_order();
    tc13_concurrent();
    tc14_builtin_entries();

    printf(g_failures ? "UNIT-REGISTRY: %d failure(s)\n"
                      : "UNIT-REGISTRY: all passed\n",
            g_failures);
    return g_failures ? 1 : 0;
}
