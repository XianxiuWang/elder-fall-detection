/**
 * test_deploy_duos.c — Verification test for Duo S deployment
 *
 * Compares C feature extraction + RandomForest output against
 * embedded Python reference data (test_data.h).
 *
 * This is a lightweight version of test_deploy.c optimized
 * for running directly on Milk-V Duo S to verify correctness.
 */
#include "../huling_deploy.h"
#include "../scaler_params.h"
#include "../test_data.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static int total_errors = 0;

static int test_single_case(int case_idx) {
    printf("\n--- Case %d: %s ---\n", case_idx, test_names[case_idx]);

    /* Build landmarks */
    PoseLandmarks lm;
    for (int k = 0; k < 33; k++) {
        lm.kp[k].x = test_landmarks[case_idx][k][0];
        lm.kp[k].y = test_landmarks[case_idx][k][1];
        lm.kp[k].z = test_landmarks[case_idx][k][2];
        lm.kp[k].visibility = test_landmarks[case_idx][k][3];
    }

    /* Extract features */
    FeatureVector fv;
    huling_extract_features(&lm, &fv);

    /* Compare full 98-dim vector */
    int case_errs = 0;
    double max_diff = 0.0;
    for (int i = 0; i < 98; i++) {
        double d = fabs((double)fv.f[i] - py_features_raw[case_idx][i]);
        if (d > 1e-4) {  /* Relaxed threshold for float vs double */
            case_errs++;
            if (d > max_diff) max_diff = d;
        }
    }

    if (case_errs > 0) {
        printf("  Features: %d/98 diffs (max diff=%.2e)\n", case_errs, max_diff);
    } else {
        printf("  Features: OK (98/98 match)\n");
    }

    /* Standardize + predict */
    standardize(fv.f);
    int c_class = predict_pose_class(fv.f);

    printf("  C predict: %d (%s)  Py predict: %d\n",
           c_class,
           c_class >= 0 && c_class < 6 ?
               (const char*[]){"walking","sitting","lying","long_sit","abnormal","fall"}[c_class]
               : "?",
           py_pred_class[case_idx]);

    if (c_class != py_pred_class[case_idx]) {
        printf("  *** CLASS MISMATCH! ***\n");
        case_errs++;
    }

    /* Compare normalized features */
    int norm_errs = 0;
    double max_norm_diff = 0.0;
    for (int i = 0; i < 98; i++) {
        double d = fabs((double)fv.f[i] - py_features_norm[case_idx][i]);
        if (d > 1e-4) {
            norm_errs++;
            if (d > max_norm_diff) max_norm_diff = d;
        }
    }
    if (norm_errs > 0) {
        printf("  Normalized: %d/98 diffs (max diff=%.2e)\n", norm_errs, max_norm_diff);
    } else {
        printf("  Normalized: OK (98/98 match)\n");
    }

    total_errors += case_errs + norm_errs;
    return case_errs > 0 || norm_errs > 0;
}

int main(void) {
    printf("========================================\n");
    printf("  HuLing C Deployment Verification\n");
    printf("  Target: Milk-V Duo S\n");
    printf("========================================\n");
    printf("  Test cases: %d\n", N_TEST_CASES);
    printf("  Features: %d\n", N_FEATURES);

    for (int c = 0; c < N_TEST_CASES; c++) {
        test_single_case(c);
    }

    printf("\n========================================\n");
    if (total_errors == 0) {
        printf("  ALL TESTS PASSED\n");
    } else {
        printf("  %d ERROR(S) FOUND\n", total_errors);
    }
    printf("========================================\n");

    return total_errors ? 1 : 0;
}
