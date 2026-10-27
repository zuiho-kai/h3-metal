/* Random-weight F32 vs H3_VAE_BF16=1 video VAE decode comparison. */
#include "h3_video_vae.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "../rand-vae";
    int T = 7, H = 8, W = 8; size_t n = (size_t)24 * T * H * W;
    float *lat = malloc(n * sizeof(float)); srand(0);
    for (size_t i = 0; i < n; i++) {  /* Box-Muller N(0,1), fixed seed */
        double u = (rand() + 1.0) / (RAND_MAX + 2.0), v = (rand() + 1.0) / (RAND_MAX + 2.0);
        lat[i] = (float)(sqrt(-2 * log(u)) * cos(6.283185307 * v));
    }
    char err[1024]; h3_video_frames a, b;
    unsetenv("H3_VAE_BF16");
    if (!h3_video_vae_decode(dir, "h3_shaders.metal", lat, T, H, W, NULL, NULL, &a, err, sizeof err)) { fprintf(stderr, "f32: %s\n", err); return 1; }
    setenv("H3_VAE_BF16", "1", 1);
    if (!h3_video_vae_decode(dir, "h3_shaders.metal", lat, T, H, W, NULL, NULL, &b, err, sizeof err)) { fprintf(stderr, "bf16: %s\n", err); return 1; }
    size_t m = (size_t)a.frames * a.height * a.width * 3;
    double mx = 0, se = 0, mean = 0, var = 0; size_t sat = 0;
    for (size_t i = 0; i < m; i++) {
        double d = (double)a.rgb[i] - b.rgb[i]; if (fabs(d) > mx) mx = fabs(d); se += d * d;
        mean += a.rgb[i]; if (a.rgb[i] <= 0.0f || a.rgb[i] >= 1.0f) sat++;
    }
    mean /= m; for (size_t i = 0; i < m; i++) var += (a.rgb[i] - mean) * (a.rgb[i] - mean); var /= m;
    double mse = se / m;
    printf("shape %dx%dx%d  f32 mean %.4f std %.4f saturated %.2f%%\n", a.frames, a.height, a.width, mean, sqrt(var), 100.0 * sat / m);
    printf("max_abs_err %.6f  mse %.3e  PSNR(peak1) %.2f dB\n", mx, mse, mse > 0 ? 10 * log10(1.0 / mse) : INFINITY);
    return 0;
}
