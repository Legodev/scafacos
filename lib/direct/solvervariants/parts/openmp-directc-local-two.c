static void
#ifdef FCS_ENABLE_OFFLOADING
#ifdef __INTEL_COMPILER
__attribute__((target(mic)))
#endif
#endif
directc_local_two(fcs_int n0, fcs_float *xyz0, fcs_float *q0, fcs_int n1, fcs_float *xyz1, fcs_float *q1, fcs_float *f,
                  fcs_float *p, fcs_float cutoff) {
    fcs_int i, j;
    fcs_float dx, dy, dz, ir;
    fcs_float p_sum, f_sum_zero, f_sum_one, f_sum_two;

#ifdef __MIC__
    __assume_aligned(xyz0, 64);
  __assume_aligned(q0, 64);
  __assume_aligned(xyz1, 64);
  __assume_aligned(q1, 64);
  __assume_aligned(f, 64);
  __assume_aligned(p, 64);
#endif

    // use static, more optimized code if cutoff is 0 (or at least very close to)
    if (fcs_fabs(cutoff) < 0.000000001) {
#pragma omp parallel for schedule(static) private(i, j, dx, dy, dz, ir, p_sum, f_sum_zero, f_sum_one, f_sum_two) firstprivate(q1, xyz0, xyz1, cutoff) shared(p, f)
        for (i = 0; i < n0; ++i) {
            p_sum = 0.0;
            f_sum_zero = 0.0;
            f_sum_one = 0.0;
            f_sum_two = 0.0;

#pragma omp parallel for schedule(static) private(j, dx, dy, dz, ir) reduction(+:p_sum, f_sum_zero, f_sum_one, f_sum_two) firstprivate(xyz0, xyz1, cutoff)
            for (j = 0; j < n1; ++j) {
                dx = xyz0[i * 3 + 0] - xyz1[j * 3 + 0];
                dy = xyz0[i * 3 + 1] - xyz1[j * 3 + 1];
                dz = xyz0[i * 3 + 2] - xyz1[j * 3 + 2];

                ir = 1.0 / fcs_sqrt(z_sqr(dx) + z_sqr(dy) + z_sqr(dz));

                fcs_float temptest = q1[j] * ir;
                p_sum += temptest;

                temptest *= ir * ir;
                f_sum_zero += temptest * dx;
                f_sum_one += temptest * dy;
                f_sum_two += temptest * dz;
            }

#pragma omp critical
            {
                p[i] += p_sum;

                f[i * 3 + 0] += f_sum_zero;
                f[i * 3 + 1] += f_sum_one;
                f[i * 3 + 2] += f_sum_two;
            }
        }
    } else {
        cutoff = 1.0 / cutoff;

#pragma omp parallel for schedule(static) private(i, j, dx, dy, dz, ir, p_sum, f_sum_zero, f_sum_one, f_sum_two) firstprivate(q1, xyz0, xyz1, cutoff) shared(p, f)
        for (i = 0; i < n0; ++i) {
            p_sum = 0.0;
            f_sum_zero = 0.0;
            f_sum_one = 0.0;
            f_sum_two = 0.0;

#pragma omp parallel for schedule(static) private(j, dx, dy, dz, ir) reduction(+:p_sum, f_sum_zero, f_sum_one, f_sum_two) firstprivate(xyz0, xyz1, cutoff)
            for (j = 0; j < n1; ++j) {
                dx = xyz0[i * 3 + 0] - xyz1[j * 3 + 0];
                dy = xyz0[i * 3 + 1] - xyz1[j * 3 + 1];
                dz = xyz0[i * 3 + 2] - xyz1[j * 3 + 2];

                ir = 1.0 / fcs_sqrt(z_sqr(dx) + z_sqr(dy) + z_sqr(dz));

                if ((cutoff > 0 && cutoff > ir) || (cutoff < 0 && -cutoff < ir))
                    continue;

                p_sum += q1[j] * ir;

                f_sum_zero += q1[j] * dx * ir * ir * ir;
                f_sum_one += q1[j] * dy * ir * ir * ir;
                f_sum_two += q1[j] * dz * ir * ir * ir;
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
}