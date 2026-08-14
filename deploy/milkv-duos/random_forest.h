#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

// 6-class pose classification
//  0=walking, 1=sitting, 2=lying, 3=long_sit, 4=abnormal, 5=fall

// Predict class (0-5) from a float feature array of length N_FEATURES
int predict_pose_class(const float *features);

// m2cgen-exported raw function (uses double internally)
// NOTE: m2cgen generates double* parameters — wrapper converts float→double
void predict_pose(double *input, double *output);

#endif
