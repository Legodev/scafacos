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
    // required for fallback code
    fcs_float p_sum, f_sum_zero, f_sum_one, f_sum_two;

    unsigned fcs_int
    roundsize = (2 * periodic[0] + 1) * (2 * periodic[1] + 1) * (2 * periodic[2] + 1) - 1;
    unsigned fcs_int
    roundsizeremainder = roundsize % 8;
    unsigned fcs_int
    roundsizefull = roundsize - roundsizeremainder;
    unsigned fcs_int
    roundpos = 0;

#ifdef DEBUG
    printf("roundsize: %d, roundsizeremainder: %d, roundsizefull: %d\n", roundsize, roundsizeremainder, roundsizefull);
#endif

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

#pragma omp parallel for schedule(static) private(i, j, pd_x, pd_y, pd_z, dx, dy, dz, ir, roundpos) firstprivate(q1, xyz0, xyz1, box_a, box_b, box_c, cutoff, roundsize, p, f, pd_x_array, pd_y_array, pd_z_array)
            for (i = 0; i < n0; ++i) {
                __m512d m512_p_sum = _mm512_setzero_pd();
                __m512d m512_f_sum_zero = _mm512_setzero_pd();
                __m512d m512_f_sum_one = _mm512_setzero_pd();
                __m512d m512_f_sum_two = _mm512_setzero_pd();

                {
#pragma omp declare reduction (mm512_add_pd : __m512d : omp_out = _mm512_add_pd(omp_out, omp_in)) // initializer(omp_priv = _mm512_setzero_pd())
#pragma omp parallel for schedule(static) private(j, pd_x, pd_y, pd_z, dx, dy, dz, ir, roundpos) reduction(mm512_add_pd:m512_p_sum, m512_f_sum_zero, m512_f_sum_one, m512_f_sum_two) firstprivate(q1, xyz0, xyz1, box_a, box_b, box_c, cutoff, roundsize, pd_x_array, pd_y_array, pd_z_array)
                    for (j = 0; j < n1; ++j) {
                        __m512d m512_xyz0_array = _mm512_set1_pd(xyz0[i * 3 + 0] - xyz1[j * 3 + 0]);
                        __m512d m512_xyz1_array = _mm512_set1_pd(xyz0[i * 3 + 1] - xyz1[j * 3 + 1]);
                        __m512d m512_xyz2_array = _mm512_set1_pd(xyz0[i * 3 + 2] - xyz1[j * 3 + 2]);

                        __m512d m512_box_a0_array = _mm512_set1_pd(box_a[0]);
                        __m512d m512_box_a1_array = _mm512_set1_pd(box_a[1]);
                        __m512d m512_box_a2_array = _mm512_set1_pd(box_a[2]);

                        __m512d m512_box_b0_array = _mm512_set1_pd(box_b[0]);
                        __m512d m512_box_b1_array = _mm512_set1_pd(box_b[1]);
                        __m512d m512_box_b2_array = _mm512_set1_pd(box_b[2]);

                        __m512d m512_box_c0_array = _mm512_set1_pd(box_c[0]);
                        __m512d m512_box_c1_array = _mm512_set1_pd(box_c[1]);
                        __m512d m512_box_c2_array = _mm512_set1_pd(box_c[2]);

                        __m512d m512_q_array = _mm512_set1_pd(q1[j]);

                        for (roundpos = 0; roundpos < roundsizefull; roundpos += 8) {
                            __m512d m512_pd_dx_array = _mm512_load_pd(&pd_x_array[roundpos]);
                            __m512d m512_pd_dy_array = _mm512_load_pd(&pd_y_array[roundpos]);
                            __m512d m512_pd_dz_array = _mm512_load_pd(&pd_z_array[roundpos]);

                            // dx_array
                            __m512d m512_dx_array = _mm512_fnmadd_pd(m512_pd_dx_array, m512_box_a0_array,
                                                                     m512_xyz0_array);
                            m512_dx_array = _mm512_fnmadd_pd(m512_pd_dy_array, m512_box_b0_array, m512_dx_array);
                            m512_dx_array = _mm512_fnmadd_pd(m512_pd_dz_array, m512_box_c0_array, m512_dx_array);

                            // dy_array
                            __m512d m512_dy_array = _mm512_fnmadd_pd(m512_pd_dx_array, m512_box_a1_array,
                                                                     m512_xyz1_array);
                            m512_dy_array = _mm512_fnmadd_pd(m512_pd_dy_array, m512_box_b1_array, m512_dy_array);
                            m512_dy_array = _mm512_fnmadd_pd(m512_pd_dz_array, m512_box_c1_array, m512_dy_array);

                            // dz_array
                            __m512d m512_dz_array = _mm512_fnmadd_pd(m512_pd_dx_array, m512_box_a2_array,
                                                                     m512_xyz2_array);
                            m512_dz_array = _mm512_fnmadd_pd(m512_pd_dy_array, m512_box_b2_array, m512_dz_array);
                            m512_dz_array = _mm512_fnmadd_pd(m512_pd_dz_array, m512_box_c2_array, m512_dz_array);

                            __m512d m512_ir_array = _mm512_mul_pd(m512_dx_array, m512_dx_array);
                            m512_ir_array = _mm512_fmadd_pd(m512_dy_array, m512_dy_array, m512_ir_array);
                            m512_ir_array = _mm512_fmadd_pd(m512_dz_array, m512_dz_array, m512_ir_array);
                            m512_ir_array = _mm512_invsqrt_pd(m512_ir_array);

                            __m512d m512_ir_tmp = _mm512_mul_pd(m512_q_array, m512_ir_array);
                            __m512d m512_ir_ir_ir_tmp = _mm512_mul_pd(m512_ir_array,
                                                                      _mm512_mul_pd(m512_ir_array, m512_ir_tmp));

                            m512_p_sum = _mm512_add_pd(m512_p_sum, m512_ir_tmp);
                            m512_f_sum_zero = _mm512_add_pd(m512_f_sum_zero, _mm512_mul_pd(m512_ir_ir_ir_tmp,
                                                                                           m512_dx_array));
                            m512_f_sum_one = _mm512_add_pd(m512_f_sum_one, _mm512_mul_pd(m512_ir_ir_ir_tmp,
                                                                                         m512_dy_array));
                            m512_f_sum_two = _mm512_add_pd(m512_f_sum_two, _mm512_mul_pd(m512_ir_ir_ir_tmp,
                                                                                         m512_dz_array));
                        }

                        if (roundsizeremainder > 0) {
                            __mmask8 _k_mask = (1 << roundsizeremainder) - 1;
                            __m512d _zeros = _mm512_setzero_pd();

                            __m512d m512_pd_dx_array = _mm512_mask_load_pd(_zeros, _k_mask, &pd_x_array[roundpos]);
                            __m512d m512_pd_dy_array = _mm512_mask_load_pd(_zeros, _k_mask, &pd_y_array[roundpos]);
                            __m512d m512_pd_dz_array = _mm512_mask_load_pd(_zeros, _k_mask, &pd_z_array[roundpos]);

                            // dx_array
                            __m512d m512_dx_array = _mm512_fnmadd_pd(m512_pd_dx_array, m512_box_a0_array,
                                                                     m512_xyz0_array);
                            m512_dx_array = _mm512_fnmadd_pd(m512_pd_dy_array, m512_box_b0_array, m512_dx_array);
                            m512_dx_array = _mm512_fnmadd_pd(m512_pd_dz_array, m512_box_c0_array, m512_dx_array);

                            // dy_array
                            __m512d m512_dy_array = _mm512_fnmadd_pd(m512_pd_dx_array, m512_box_a1_array,
                                                                     m512_xyz1_array);
                            m512_dy_array = _mm512_fnmadd_pd(m512_pd_dy_array, m512_box_b1_array, m512_dy_array);
                            m512_dy_array = _mm512_fnmadd_pd(m512_pd_dz_array, m512_box_c1_array, m512_dy_array);

                            // dz_array
                            __m512d m512_dz_array = _mm512_fnmadd_pd(m512_pd_dx_array, m512_box_a2_array,
                                                                     m512_xyz2_array);
                            m512_dz_array = _mm512_fnmadd_pd(m512_pd_dy_array, m512_box_b2_array, m512_dz_array);
                            m512_dz_array = _mm512_fnmadd_pd(m512_pd_dz_array, m512_box_c2_array, m512_dz_array);

                            __m512d m512_ir_array = _mm512_mul_pd(m512_dx_array, m512_dx_array);
                            m512_ir_array = _mm512_fmadd_pd(m512_dy_array, m512_dy_array, m512_ir_array);
                            m512_ir_array = _mm512_fmadd_pd(m512_dz_array, m512_dz_array, m512_ir_array);
                            m512_ir_array = _mm512_invsqrt_pd(m512_ir_array);

                            __m512d m512_ir_tmp = _mm512_mul_pd(m512_q_array, m512_ir_array);
                            __m512d m512_ir_ir_ir_tmp = _mm512_mul_pd(m512_ir_array,
                                                                      _mm512_mul_pd(m512_ir_array, m512_ir_tmp));

                            m512_p_sum = _mm512_mask_add_pd(m512_p_sum, _k_mask, m512_p_sum, m512_ir_tmp);
                            m512_f_sum_zero = _mm512_mask_add_pd(m512_f_sum_zero, _k_mask, m512_f_sum_zero,
                                                                 _mm512_mul_pd(m512_ir_ir_ir_tmp, m512_dx_array));
                            m512_f_sum_one = _mm512_mask_add_pd(m512_f_sum_one, _k_mask, m512_f_sum_one,
                                                                _mm512_mul_pd(m512_ir_ir_ir_tmp, m512_dy_array));
                            m512_f_sum_two = _mm512_mask_add_pd(m512_f_sum_two, _k_mask, m512_f_sum_two,
                                                                _mm512_mul_pd(m512_ir_ir_ir_tmp, m512_dz_array));
                        }
                    }
                }

                p[i] += _mm512_reduce_add_pd(m512_p_sum);
                f[i * 3 + 0] += _mm512_reduce_add_pd(m512_f_sum_zero);
                f[i * 3 + 1] += _mm512_reduce_add_pd(m512_f_sum_one);
                f[i * 3 + 2] += _mm512_reduce_add_pd(m512_f_sum_two);
            }
        }
// less optimized but dynamic fallback code
#include "openmp-directc_local_periodic-fallback.c"
    }
}

