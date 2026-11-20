#include <math.h>
#include <stdint.h>

static inline float cross(float ax, float ay, float bx, float by) {
    return ax * by - ay * bx;
}

/* Orthographic depth in millimeters, with the volume origin at grid center. */
void rasterize(const float *vertices, const int32_t *indices, int count, int size,
               float spacing, float *zbuffer) {
    const float center = (size - 1) * 0.5f;
    for (int face = 0; face < count; face++) {
        const float *a = vertices + 3 * indices[3 * face];
        const float *b = vertices + 3 * indices[3 * face + 1];
        const float *c = vertices + 3 * indices[3 * face + 2];
        const float bx = b[0] - a[0], by = b[1] - a[1], cx = c[0] - a[0], cy = c[1] - a[1];
        const float area = cross(bx, by, cx, cy);
        if (fabsf(area) < 1e-6f) continue;
        int u0 = fmaxf(0.0f, ceilf(fminf(a[0], fminf(b[0], c[0])) / spacing + center));
        int v0 = fmaxf(0.0f, ceilf(fminf(a[1], fminf(b[1], c[1])) / spacing + center));
        int u1 = fminf(size - 1, floorf(fmaxf(a[0], fmaxf(b[0], c[0])) / spacing + center));
        int v1 = fminf(size - 1, floorf(fmaxf(a[1], fmaxf(b[1], c[1])) / spacing + center));
        for (int v = v0; v <= v1; v++) {
            float y = (v - center) * spacing - a[1];
            for (int u = u0; u <= u1; u++) {
                const float x = (u - center) * spacing - a[0];
                const float beta = cross(x, y, cx, cy) / area;
                const float gamma = cross(bx, by, x, y) / area;
                if (beta < -1e-5f || gamma < -1e-5f || beta + gamma > 1.00001f) continue;
                const float z = a[2] + beta * (b[2] - a[2]) + gamma * (c[2] - a[2]);
                if (z < zbuffer[v * size + u]) zbuffer[v * size + u] = z;
            }
        }
    }
}

static inline float sensor_depth(const float *depth, int width, int height, int x, int y) {
    return (unsigned)x < (unsigned)width && (unsigned)y < (unsigned)height
           ? depth[y * width + x] : 0.0f;
}

/* One coarse CAD fit per camera-grid position. XYZ samples are in millimeters. */
void scan_depth(const float *depth, const float *median, int width, int height,
                float fx, float fy, const float *surface, const float *boundary,
                int orientations, int n, int rim, int step, float *fits) {
    const float shifts[] = {-18.f, -9.f, 0.f, 9.f, 18.f};
    int cell = 0;
    for (int y = step / 2; y < height; y += step) {
        for (int x = step / 2; x < width; x += step, cell++) {
            const float z = median[y * width + x];
            if (z < 250.f || z > 1800.f) continue;
            const float sx = fx / z, sy = fy / z;
            float *result = fits + cell * 4;
            for (int a = 0; a < orientations; a++) {
                const float *p = surface + 3 * n * a;
                float evidence[5] = {0};
                for (int i = 0; i < n; i++, p += 3) {
                    const int u = (int)floorf(x + p[0] * sx + .5f);
                    const int v = (int)floorf(y + p[1] * sy + .5f);
                    const float observed = sensor_depth(depth, width, height, u, v);
                    if (!observed) continue;
                    const float error = observed - z - p[2];
                    for (int s = 0; s < 5; s++) {
                        const float diff = error - shifts[s];
                        evidence[s] += fmaxf(0.f, 1.f - fabsf(diff) / 11.f) - (diff > 11.f ? .26f : 0.f);
                    }
                }
                const float *q = boundary + 3 * rim * a;
                for (int i = 0; i < rim; i++, q += 3) {
                    const int u = (int)floorf(x + q[0] * sx + .5f);
                    const int v = (int)floorf(y + q[1] * sy + .5f);
                    const float error = sensor_depth(depth, width, height, u, v) - z - q[2];
                    for (int s = 0; s < 5; s++) {
                        if (fabsf(error - shifts[s]) < 8.f) evidence[s] -= .32f * n / rim;
                    }
                }
                for (int s = 0; s < 5; s++) {
                    const float fit = evidence[s] / n;
                    if (fit <= result[0]) continue;
                    result[0] = fit;
                    result[1] = a;
                    result[2] = z;
                    result[3] = shifts[s];
                }
            }
        }
    }
}
