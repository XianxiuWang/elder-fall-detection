/**
 * main.c — HuLing standalone demo for Milk-V Duo S
 *
 * Reads 33 keypoints from stdin (or file), runs inference, prints result.
 *
 * Input format (JSON lines):
 *   {"x":0.5,"y":0.3,"z":0.0,"v":0.95}
 *   ... (33 lines, one per keypoint)
 *   --- (separator)
 *
 * Usage on Duo S:
 *   ./huling_demo < test_keypoints.txt
 *
 * Or with real-time network input:
 *   nc -l 8888 | ./huling_demo
 */
#include "../huling_deploy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char *argv[]) {
    printf("========================================\n");
    printf("  HuLing Pose Classifier for Duo S\n");
    printf("  Model: RandomForest 100-tree, 98-dim\n");
    printf("  Classes: walking/sitting/lying/long_sit/abnormal/fall\n");
    printf("========================================\n\n");

    int frame_count = 0;
    int class_counts[6] = {0};
    clock_t total_start = clock();

    while (1) {
        PoseLandmarks lm;
        int kp_read = 0;

        /* Read 33 keypoints from stdin */
        for (int i = 0; i < 33; i++) {
            float x, y, z, v;
            int n = scanf("{\"x\":%f,\"y\":%f,\"z\":%f,\"v\":%f}", &x, &y, &z, &v);
            if (n != 4) {
                /* Try simpler format: just 4 floats per line */
                n = scanf("%f %f %f %f", &x, &y, &z, &v);
                if (n != 4) {
                    goto done;
                }
            }
            /* Consume newline */
            while (getchar() != '\n' && !feof(stdin)) {}

            lm.kp[i].x = x;
            lm.kp[i].y = y;
            lm.kp[i].z = z;
            lm.kp[i].visibility = v;
            kp_read++;
        }

        if (kp_read != 33) {
            printf("Error: expected 33 keypoints, got %d\n", kp_read);
            break;
        }

        /* Consume separator line (---) */
        {
            char sep[16];
            if (fgets(sep, sizeof(sep), stdin) == NULL && feof(stdin)) {
                /* Last frame, no separator needed */
            }
        }

        /* Run prediction */
        PosePrediction result;
        huling_predict(&lm, &result);

        printf("[Frame %4d] %10s  conf=%.3f  (",
               frame_count,
               result.class_name,
               result.confidence);

        for (int c = 0; c < HULING_N_CLASSES; c++) {
            printf("%.2f%s", result.scores[c],
                   c < HULING_N_CLASSES - 1 ? " " : "");
        }
        printf(")\n");

        class_counts[result.class_id]++;
        frame_count++;

        /* Check for EOF */
        int peek = getchar();
        if (peek == EOF) break;
        ungetc(peek, stdin);
    }

done:
    clock_t total_end = clock();
    double total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;

    printf("\n========================================\n");
    printf("  Summary\n");
    printf("========================================\n");
    printf("  Total frames:   %d\n", frame_count);
    printf("  Total time:     %.3f s\n", total_time);
    if (frame_count > 0) {
        printf("  Avg inference:  %.2f ms/frame\n",
               total_time * 1000.0 / frame_count);
        printf("  FPS:            %.1f\n",
               frame_count / total_time);
        printf("\n  Class distribution:\n");
        for (int c = 0; c < HULING_N_CLASSES; c++) {
            printf("    %-10s: %4d (%5.1f%%)\n",
                   HULING_CLASS_NAMES[c],
                   class_counts[c],
                   frame_count > 0 ? class_counts[c] * 100.0 / frame_count : 0.0);
        }
    }

    return 0;
}
