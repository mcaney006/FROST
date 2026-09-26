// FROST boundary proof: MLX runs a real matmul on the Apple GPU via mlx-c,
// using the Homebrew prebuilt metallib (no Metal source compiler required).
#include <stdio.h>
#include <math.h>
#include "mlx/c/mlx.h"

int main(void) {
    mlx_device gpu = mlx_device_new_type(MLX_GPU, 0);
    bool avail = false;
    mlx_device_is_available(&avail, gpu);
    mlx_string ds = mlx_string_new();
    mlx_device_tostring(&ds, gpu);
    printf("gpu_available=%d device=%s\n", avail, mlx_string_data(ds));
    if (!avail) { printf("RESULT=FAIL no gpu\n"); return 2; }

    mlx_stream s = mlx_default_gpu_stream_new();

    // A [2x3], B [3x2]
    float a[6] = {1,2,3, 4,5,6};
    float b[6] = {1,0, 0,1, 1,1};
    int as[2] = {2,3}, bs[2] = {3,2};
    mlx_array A = mlx_array_new_data(a, as, 2, MLX_FLOAT32);
    mlx_array B = mlx_array_new_data(b, bs, 2, MLX_FLOAT32);
    mlx_array C = mlx_array_new();
    mlx_matmul(&C, A, B, s);
    mlx_array_eval(C);   // force GPU execution

    const float *cd = mlx_array_data_float32(C);
    // reference: [[1+3, 2+3],[4+6,5+6]] = [[4,5],[10,11]]
    float ref[4] = {4,5,10,11};
    double err = 0; for (int i=0;i<4;i++) err += fabs(cd[i]-ref[i]);
    printf("C = [%.1f %.1f %.1f %.1f]  abs_err=%.6f\n", cd[0],cd[1],cd[2],cd[3], err);
    printf("RESULT=%s\n", err < 1e-5 ? "PASS" : "FAIL");

    mlx_array_free(A); mlx_array_free(B); mlx_array_free(C);
    mlx_stream_free(s); mlx_device_free(gpu); mlx_string_free(ds);
    return err < 1e-5 ? 0 : 1;
}
