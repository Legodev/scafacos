static void
#ifdef FCS_ENABLE_OFFLOADING
#ifdef __INTEL_COMPILER
__attribute__((target(mic)))
#endif
#endif
directc_local_periodic(fcs_int n0, fcs_float *xyz0, fcs_float *q0, fcs_int n1, fcs_float *xyz1, fcs_float *q1,
                       fcs_float *f, fcs_float *p, fcs_int *periodic, fcs_float *box_a, fcs_float *box_b,
                       fcs_float *box_c, fcs_float cutoff) {

#ifdef __MIC__
    __assume_aligned(xyz0, 64);
  __assume_aligned(q0, 64);
  __assume_aligned(xyz1, 64);
  __assume_aligned(q1, 64);
  __assume_aligned(f, 64);
  __assume_aligned(p, 64);
  __assume_aligned(box_a, 64);
  __assume_aligned(box_b, 64);
  __assume_aligned(box_c, 64);
#endif

    fcs_int i, j, pd_x, pd_y, pd_z;
    fcs_float dx, dy, dz, ir;
    fcs_float p_sum, f_sum_zero, f_sum_one, f_sum_two;

    unsigned fcs_int
    roundsize = (2 * periodic[0] + 1) * (2 * periodic[1] + 1) * (2 * periodic[2] + 1) - 1;
    unsigned fcs_int
    roundpos = 0;

// don't try to calculate the periodicity if system is nonperiodic
    if (roundsize > 0) {
// use static, more optimized code if cutoff is 0 (or at least very close to) and
// the periodicity is less or equal (1, 1, 1)
        if (fcs_fabs(cutoff) < 0.000000001 && roundsize < 32) {
            fcs_float pd_x_array[32] __attribute__((aligned(64)));
            fcs_float pd_y_array[32] __attribute__((aligned(64)));
            fcs_float pd_z_array[32] __attribute__((aligned(64)));

            for (pd_x = -periodic[0]; pd_x <= periodic[0]; ++pd_x)
                for (pd_y = -periodic[1]; pd_y <= periodic[1]; ++pd_y)
                    for (pd_z = -periodic[2]; pd_z <= periodic[2]; ++pd_z) {
                        if (pd_x == 0 && pd_y == 0 && pd_z == 0)
                            continue;

                        pd_x_array[roundpos] = pd_x;
                        pd_y_array[roundpos] = pd_y;
                        pd_z_array[roundpos] = pd_z;

#ifdef PRINT_PARTICLES
#ifndef __MIC__
                        printf("%f - %f - %f - %d - %d\n", (float)pd_x_array[roundpos], (float)pd_y_array[roundpos], (float)pd_z_array[roundpos], roundpos, roundsize);
#endif
#endif

                        roundpos++;
                    }

#pragma omp parallel for schedule(static) private(i, j, pd_x, pd_y, pd_z, dx, dy, dz, ir, roundpos, p_sum, f_sum_zero, f_sum_one, f_sum_two) firstprivate(q1, xyz0, xyz1, box_a, box_b, box_c, cutoff, roundsize) shared(p, f, pd_x_array, pd_y_array, pd_z_array)
            for (i = 0; i < n0; ++i) {
                p_sum = 0.0;
                f_sum_zero = 0.0;
                f_sum_one = 0.0;
                f_sum_two = 0.0;

#pragma omp parallel num_threads(4)
                {
#pragma omp parallel for schedule(static) private(j, pd_x, pd_y, pd_z, dx, dy, dz, ir, roundpos) reduction(+:p_sum, f_sum_zero, f_sum_one, f_sum_two) firstprivate(q1, xyz0, xyz1, box_a, box_b, box_c, cutoff, roundsize, pd_x_array, pd_y_array, pd_z_array)
                    for (j = 0; j < n1; ++j) {
                        for (roundpos = 0; roundpos < roundsize; roundpos++) {
                            dx = xyz0[i * 3 + 0] - xyz1[j * 3 + 0]
                                 - (pd_x_array[roundpos] * box_a[0])
                                 - (pd_y_array[roundpos] * box_b[0])
                                 - (pd_z_array[roundpos] * box_c[0]);
                            dy = xyz0[i * 3 + 1] - xyz1[j * 3 + 1]
                                 - (pd_x_array[roundpos] * box_a[1])
                                 - (pd_y_array[roundpos] * box_b[1])
                                 - (pd_z_array[roundpos] * box_c[1]);
                            dz = xyz0[i * 3 + 2] - xyz1[j * 3 + 2]
                                 - (pd_x_array[roundpos] * box_a[2])
                                 - (pd_y_array[roundpos] * box_b[2])
                                 - (pd_z_array[roundpos] * box_c[2]);

                            ir = 1.0 / fcs_sqrt(z_sqr(dx) + z_sqr(dy) + z_sqr(dz));

                            fcs_float temptest = q1[j] * ir;
                            p_sum += temptest;

                            temptest *= ir * ir;
                            f_sum_zero += temptest * dx;
                            f_sum_one += temptest * dy;
                            f_sum_two += temptest * dz;
                        }
                    }
                }

#pragma omp critical
                {
                    p[i] += p_sum;

                    f[i * 3 + 0] += f_sum_zero;
                    f[i * 3 + 1] += f_sum_one;
                    f[i * 3 + 2] += f_sum_two;
                }
            }
        }
// less optimized but dynamic fallback code
#include "openmp-directc_local_periodic-fallback.c"
    }
}

