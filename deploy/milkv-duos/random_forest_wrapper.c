// Wrapper: standardize -> classify -> return class
// Fixed: m2cgen uses double* internally, so convert float→double→float
// Model trained on 3 classes: [0, 1, 5] → m2cgen outputs 3 scores
#include "scaler_params.h"
#include "random_forest.h"
#include <string.h>
#include <stdio.h>

#define N_MODEL_CLASSES 3
/* Map model output index → actual class ID */
static const int MODEL_CLASS_MAP[N_MODEL_CLASSES] = {0, 1, 5};
/* All 6 class names (for display) */
static const char *CLASS_NAMES[6] = {
    "walking", "sitting", "lying", "long_sit", "abnormal", "fall"
};

int predict_pose_class(const float *features) {
    float norm_features[N_FEATURES];
    memcpy(norm_features, features, sizeof(float) * N_FEATURES);
    standardize(norm_features);
    
    // Convert float→double for m2cgen (which uses double internally)
    double input_dbl[N_FEATURES];
    for (int i = 0; i < N_FEATURES; i++) {
        input_dbl[i] = (double)norm_features[i];
    }
    
    // m2cgen outputs 3 scores (for classes [0, 1, 5])
    double scores_dbl[N_MODEL_CLASSES];
    predict_pose(scores_dbl, input_dbl);
    
    // argmax over model's 3 classes
    int best_idx = 0;
    double best_score = scores_dbl[0];
    for (int i = 1; i < N_MODEL_CLASSES; i++) {
        if (scores_dbl[i] > best_score) {
            best_score = scores_dbl[i];
            best_idx = i;
        }
    }
    return MODEL_CLASS_MAP[best_idx];
}

const char* class_name(int cls) {
    return (cls >= 0 && cls < 6) ? CLASS_NAMES[cls] : "unknown";
}
