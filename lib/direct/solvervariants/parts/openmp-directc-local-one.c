static void
#ifdef FCS_ENABLE_OFFLOADING
#ifdef __INTEL_COMPILER
__attribute__((target(mic)))
#endif
#endif
directc_local_one(fcs_int nout, fcs_int nin, fcs_float *xyz, fcs_float *q, fcs_float *f, fcs_float *p,
                  fcs_float cutoff) {
    fcs_int i, j;
    fcs_float dx, dy, dz, ir;
    fcs_float p_sum, f_sum_zero, f_sum_one, f_sum_two;

#ifdef __MIC__
    __assume_aligned(xyz, 64);
  __assume_aligned(q, 64);
  __assume_aligned(f, 64);
  __assume_aligned(p, 64);
#endif

    // use static, more optimized code if cutoff is 0 (or at least very close to)
    if (fcs_fabs(cutoff) < 0.000000001) {

        // not parallelizable because of access conflicts where inner loop accesses the field of the next outer loop
        for (i = 0; i < nout; ++i) {
            p_sum = 0.0;
            f_sum_zero = 0.0;
            f_sum_one = 0.0;
            f_sum_two = 0.0;

            // parallelizable
#pragma omp parallel for schedule(static) private(j, dx, dy, dz, ir) firstprivate(i, q, xyz, cutoff) reduction(+:p_sum, f_sum_zero, f_sum_one, f_sum_two)
            for (j = i + 1; j < nout; ++j) {
                dx = xyz[i * 3 + 0] - xyz[j * 3 + 0];
                dy = xyz[i * 3 + 1] - xyz[j * 3 + 1];
                dz = xyz[i * 3 + 2] - xyz[j * 3 + 2];

                ir = 1.0 / fcs_sqrt(z_sqr(dx) + z_sqr(dy) + z_sqr(dz));

                fcs_float temptest = q[j] * ir;
                p_sum += temptest;

                temptest *= ir * ir;
                f_sum_zero += temptest * dx;
                f_sum_one += temptest * dy;
                f_sum_two += temptest * dz;

                temptest = q[i] * ir;
                p[j] += temptest;

                temptest *= ir * ir;
                f[j * 3 + 0] -= temptest * dx;
                f[j * 3 + 1] -= temptest * dy;
                f[j * 3 + 2] -= temptest * dz;
            }

#pragma omp parallel for schedule(static) private(j, dx, dy, dz, ir) firstprivate(i, q, xyz, cutoff) reduction(+:p_sum, f_sum_zero, f_sum_one, f_sum_two)
            for (j = nout; j < nin; ++j) {
                dx = xyz[i * 3 + 0] - xyz[j * 3 + 0];
                dy = xyz[i * 3 + 1] - xyz[j * 3 + 1];
                dz = xyz[i * 3 + 2] - xyz[j * 3 + 2];

                ir = 1.0 / fcs_sqrt(z_sqr(dx) + z_sqr(dy) + z_sqr(dz));

                fcs_float temptest = q[j] * ir;
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

// not parallelizable because of access conflicts where inner loop accesses the field of the next outer loop
        for (i = 0; i < nout; ++i) {
            p_sum = 0.0;
            f_sum_zero = 0.0;
            f_sum_one = 0.0;
            f_sum_two = 0.0;

// parallelizable
#pragma omp parallel for schedule(static) private(j, dx, dy, dz, ir) firstprivate(i, q, xyz, cutoff) reduction(+:p_sum, f_sum_zero, f_sum_one, f_sum_two)
            for (j = i + 1; j < nout; ++j) {
                dx = xyz[i * 3 + 0] - xyz[j * 3 + 0];
                dy = xyz[i * 3 + 1] - xyz[j * 3 + 1];
                dz = xyz[i * 3 + 2] - xyz[j * 3 + 2];

                ir = 1.0 / fcs_sqrt(z_sqr(dx) + z_sqr(dy) + z_sqr(dz));

                if ((cutoff > 0 && cutoff > ir) || (cutoff < 0 && -cutoff < ir))
                    continue;

                p_sum += q[j] * ir;
                p[j] += q[i] * ir;

                f_sum_zero += q[j] * dx * ir * ir * ir;
                f_sum_one += q[j] * dy * ir * ir * ir;
                f_sum_two += q[j] * dz * ir * ir * ir;

                f[j * 3 + 0] -= q[i] * dx * ir * ir * ir;
                f[j * 3 + 1] -= q[i] * dy * ir * ir * ir;
                f[j * 3 + 2] -= q[i] * dz * ir * ir * ir;
            }

#pragma omp parallel for schedule(static) private(j, dx, dy, dz, ir) firstprivate(i, q, xyz, cutoff) reduction(+:p_sum, f_sum_zero, f_sum_one, f_sum_two)
            for (j = nout; j < nin; ++j) {
                dx = xyz[i * 3 + 0] - xyz[j * 3 + 0];
                dy = xyz[i * 3 + 1] - xyz[j * 3 + 1];
                dz = xyz[i * 3 + 2] - xyz[j * 3 + 2];

                ir = 1.0 / fcs_sqrt(z_sqr(dx) + z_sqr(dy) + z_sqr(dz));

                if ((cutoff > 0 && cutoff > ir) || (cutoff < 0 && -cutoff < ir))
                    continue;

                p_sum += q[j] * ir;

                f_sum_zero += q[j] * dx * ir * ir * ir;
                f_sum_one += q[j] * dy * ir * ir * ir;
                f_sum_two += q[j] * dz * ir * ir * ir;
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