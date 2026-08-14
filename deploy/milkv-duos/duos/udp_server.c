/**
 * udp_server.c — Real-time inference server for Milk-V Duo S
 *
 * Listens for UDP keypoint packets from PC (keypoint_bridge.py),
 * runs HuLing prediction, prints/streams results.
 *
 * Packet format (from keypoint_bridge.py):
 *   4 bytes  frame_id (uint32 LE)
 *   528 bytes 33 keypoints × 4 floats (x,y,z,v) each
 *   Total: 532 bytes
 *
 * Build: add to CMakeLists.txt or compile manually
 *   riscv64-unknown-linux-musl-gcc -o huling_server udp_server.c
 *     ../huling_features.c ../random_forest.c ../random_forest_wrapper.c
 *     -lm -Os -Wall -I..
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <time.h>

#include "../huling_deploy.h"

#define UDP_PORT 8888
#define PACKET_SIZE 532  /* 4 + 33*4*4 */

static void print_result(uint32_t frame_id, const PosePrediction *result, double ms) {
    printf("[%5u] %10s conf=%.3f (%5.1fms)\n",
           frame_id, result->class_name, result->confidence, ms);
    fflush(stdout);
}

int main(void) {
    printf("========================================\n");
    printf("  HuLing Real-time Server for Duo S\n");
    printf("  Listening on UDP port %d\n", UDP_PORT);
    printf("========================================\n");

    /* Create UDP socket */
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("socket");
        return 1;
    }

    /* Set receive timeout (1s) for clean shutdown */
    struct timeval tv = {1, 0};
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(UDP_PORT);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(sock);
        return 1;
    }

    printf("[INFO] Ready. Waiting for keypoint packets...\n");
    printf("[INFO] Start PC: python keypoint_bridge.py --output udp --host <ip> --port %d\n\n",
           UDP_PORT);

    uint8_t buf[PACKET_SIZE];
    uint32_t frame_count = 0;
    uint32_t lost_frames = 0;
    uint32_t last_frame_id = 0;
    clock_t total_start = clock();

    while (1) {
        ssize_t n = recvfrom(sock, buf, sizeof(buf), 0, NULL, NULL);
        if (n < 0) {
            /* Timeout — check if we should exit (for now just continue) */
            continue;
        }
        if (n != PACKET_SIZE) {
            printf("[WARN] Expected %d bytes, got %zd\n", PACKET_SIZE, n);
            continue;
        }

        /* Parse frame_id */
        uint32_t frame_id;
        memcpy(&frame_id, buf, 4);

        /* Check for lost frames */
        if (frame_count > 0 && frame_id != last_frame_id + 1) {
            lost_frames += (frame_id - last_frame_id - 1);
        }
        last_frame_id = frame_id;

        /* Parse 33 keypoints */
        PoseLandmarks lm;
        float *kp_data = (float*)(buf + 4);
        for (int i = 0; i < 33; i++) {
            lm.kp[i].x = kp_data[i * 4 + 0];
            lm.kp[i].y = kp_data[i * 4 + 1];
            lm.kp[i].z = kp_data[i * 4 + 2];
            lm.kp[i].visibility = kp_data[i * 4 + 3];
        }

        /* Run inference */
        clock_t t0 = clock();
        PosePrediction result;
        huling_predict(&lm, &result);
        double ms = (double)(clock() - t0) * 1000.0 / CLOCKS_PER_SEC;

        print_result(frame_id, &result, ms);
        frame_count++;

        /* Alert on fall */
        if (result.class_id == 5 && result.confidence > 0.6) {
            printf("  *** FALL DETECTED! ***\n");
        }
    }

    clock_t total_end = clock();
    double total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;

    printf("\n========================================\n");
    printf("  Session Summary\n");
    printf("========================================\n");
    printf("  Frames received: %u\n", frame_count);
    printf("  Lost frames:     %u\n", lost_frames);
    printf("  Total time:      %.1f s\n", total_time);
    if (frame_count > 0) {
        printf("  Avg inference:   %.2f ms\n", total_time * 1000.0 / frame_count);
        printf("  Effective FPS:   %.1f\n", frame_count / total_time);
    }

    close(sock);
    return 0;
}
