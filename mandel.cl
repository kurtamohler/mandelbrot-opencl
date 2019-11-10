#define PI 3.14159265f

inline float sumOfProd(float a0, float a1, float b0, float b1) {
    float b = b0 * b1;
    float err = fma(-b0, b1, b);
    float dop = fma(a0, a1, b);

    return dop + err;
}

// mandel_buffer is a 1-D representation of a 2-D matrix
// It is organized row by row. For instance, mandel_buffer[0]
// corresponds to col 0 row 0, and mandel_buffer[1] corresponds
// to col 1 row 0.
__kernel void mandel(
    __global float* mandel_frame,
    const unsigned int x_size,
    const unsigned int y_size,
    const float x_min,
    const float x_max,
    const float y_min,
    const float y_max,
    const unsigned int max_loops
) {
    int i = get_global_id(0);

    const unsigned int total_pixels = x_size * y_size;

    if (i < total_pixels) {
        int x = i % x_size;
        int y = i / x_size;

        // Map col and row numbers to value between 0 and 1
        float x_ratio = ((float) x) / ((float) (x_size-1));
        float y_ratio = ((float) y) / ((float) (y_size-1));

        float c_real = x_min + x_ratio * (x_max - x_min);
        float c_imm = y_min + y_ratio * (y_max - y_min);

        float c_real_start = c_real;
        float c_imm_start = c_imm;

        bool diverged = false;
        float divergence_count_smoothed = 0;

        float c_real_squ;
        float c_imm_squ;
        float c_mag_squ;

        float c_imm_new;

        unsigned int loop_num = 1;

        while ((loop_num <= max_loops) && !diverged) {
            __attribute__((opencl_unroll_hint))
            for (int unroll = 0; unroll < 8; unroll++) {
                // loop
                c_mag_squ = sumOfProd(c_real, c_real, c_imm, c_imm);

                if (((c_mag_squ) > 4) && !diverged) {
                    diverged = true;
                    divergence_count_smoothed =
                        (float)loop_num - log2(0.5f * log((float)c_mag_squ));
                        // loop_num;
                }

                c_imm_new = fma(2.0f*c_real, c_imm, c_imm_start);
                c_real = fma(c_real, c_real, fma(-c_imm, c_imm, c_real_start));

                c_imm = c_imm_new;

                loop_num++;
            }
            
        }

        mandel_frame[i] = diverged ? divergence_count_smoothed : 0;
    }
}

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
} Color;

__kernel void mandel_color(
    __global float* mandel_counts,
    __global Color* mandel_colors,
    const unsigned int x_size,
    const unsigned int y_size,
    const unsigned int color_add,
    const unsigned int max_loops
) {
    int i = get_global_id(0);

    const unsigned int total_pixels = x_size * y_size;

    if (i < total_pixels) {
        int x = i % x_size;
        int y = i / x_size;

        float point_val = mandel_counts[i];

        float r_val = 0;
        float g_val = 0;
        float b_val = 0;

        float max_loops_recip = 8.0f / ((float) max_loops);

        if (point_val > 0) {
            point_val += ((float) color_add);
            r_val = 127.0f + 127.0f * sin(point_val * 2.0f * PI * max_loops_recip);
            g_val = 127.0f + 127.0f * sin(point_val * 0.02f);
            b_val = 255.0f - r_val;
        }

        mandel_colors[i].r = r_val;
        mandel_colors[i].g = g_val;
        mandel_colors[i].b = b_val;
        mandel_colors[i].a = 255;
    }
}
