





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

        bool in_set = false;
        float divergence_count_smoothed = 0;

        float c_real_squ;
        float c_imm_squ;
        float c_mag_squ;

        float c_imm_new;

        unsigned int loop_num = 1;

        while ((loop_num <= max_loops) && !in_set) {
            __attribute__((opencl_unroll_hint))
            for (int unroll = 0; unroll < 8; unroll++) {
                // loop
                c_real_squ = c_real * c_real;
                c_imm_squ = c_imm * c_imm;
                c_mag_squ = c_real_squ + c_imm_squ;

                if (((c_mag_squ) > 4) && !in_set) {
                    in_set = true;
                    divergence_count_smoothed =
                        (float)loop_num - log2(0.5f * log(c_mag_squ));
                }

                c_imm_new = c_imm_start + 2.0f*c_real*c_imm;
                c_real = c_real_start + c_real_squ - c_imm_squ;
                c_imm = c_imm_new;

                loop_num++;
            }
            
        }

        mandel_frame[i] = in_set ? divergence_count_smoothed : 0;
    }
}