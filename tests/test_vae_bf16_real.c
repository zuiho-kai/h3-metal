/* Real-weight F32 vs H3_VAE_BF16=1 video VAE decode of a dumped latent.
 * usage: h3_vae_bf16_real_test <weight_dir> <latent.bin C*T*H*W f32> T H W <out_prefix> [mode both|f32|bf16]
 * writes <out_prefix>-f32.rgb / -bf16.rgb (frames*H*W*3 f32) and prints frame shape. */
#include "h3_video_vae.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int run(const char *dir, const float *lat, int T, int H, int W, const char *path, int bf) {
    char err[1024]; h3_video_frames f;
    if (bf) setenv("H3_VAE_BF16", "1", 1); else unsetenv("H3_VAE_BF16");
    if (!h3_video_vae_decode(dir, "h3_shaders.metal", lat, T, H, W, NULL, NULL, &f, err, sizeof err)) {
        fprintf(stderr, "%s: %s\n", bf ? "bf16" : "f32", err); return 0; }
    printf("%s frames %d %dx%d\n", bf ? "bf16" : "f32", f.frames, f.height, f.width);
    FILE *o = fopen(path, "wb"); size_t m = (size_t)f.frames * f.height * f.width * 3;
    fwrite(f.rgb, sizeof(float), m, o); fclose(o); h3_video_frames_free(&f); return 1;
}
int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage\n"); return 2; }
    int T = atoi(argv[3]), H = atoi(argv[4]), W = atoi(argv[5]);
    const char *mode = argc > 7 ? argv[7] : "both";
    size_t n = (size_t)24 * T * H * W; float *lat = malloc(n * sizeof(float));
    FILE *in = fopen(argv[2], "rb"); if (!in || fread(lat, sizeof(float), n, in) != n) { fprintf(stderr, "bad latent\n"); return 1; }
    fclose(in);
    char p[2048];
    if (strcmp(mode, "bf16")) { snprintf(p, sizeof p, "%s-f32.rgb", argv[6]); if (!run(argv[1], lat, T, H, W, p, 0)) return 1; }
    if (strcmp(mode, "f32")) { snprintf(p, sizeof p, "%s-bf16.rgb", argv[6]); if (!run(argv[1], lat, T, H, W, p, 1)) return 1; }
    return 0;
}
