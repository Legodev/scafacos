// less optimized but dynamic fallback code
        if (!(fcs_fabs(cutoff) < 0.000000001 && roundsize < 32))
        {
            printf("the highly optimized code does not support cutoff or an periodicity larger then (1, 1, 1)\n");
            printf("falling back to an less optimized code. cutoff: %f\n", cutoff);

            fcs_int * pd_x_array = calloc(roundsize, sizeof(fcs_int));
            fcs_int * pd_y_array = calloc(roundsize, sizeof(fcs_int));
            fcs_int * pd_z_array = calloc(roundsize, sizeof(fcs_int));

            for (pd_x = -periodic[0]; pd_x <= periodic[0]; ++pd_x)
                for (pd_y = -periodic[1]; pd_y <= periodic[1]; ++pd_y)
                    for (pd_z = -periodic[2]; pd_z <= periodic[2]; ++pd_z)
                    {
                        if (pd_x == 0 && pd_y == 0 && pd_z == 0)
                            continue;

                        pd_x_array[roundpos] = pd_x;
                        pd_y_array[roundpos] = pd_y;
                        pd_z_array[roundpos] = pd_z;

                        roundpos++;
                    }

            if (fcs_fabs(cutoff) > 0)
                cutoff = 1.0 / cutoff;

#pragma omp parallel for schedule(static) private(i, j, pd_x, pd_y, pd_z, dx, dy, dz, ir, roundpos, p_sum, f_sum_zero, f_sum_one, f_sum_two) firstprivate(q1, xyz0, xyz1, box_a, box_b, box_c, cutoff, roundsize, pd_x_array, pd_y_array, pd_z_array) shared(p, f)
            for (i = 0; i < n0; ++i)
            {
                p_sum = 0.0;
                f_sum_zero = 0.0;
                f_sum_one = 0.0;
                f_sum_two = 0.0;

#pragma omp parallel num_threads(4)
                {
                    fcs_float * dx_array = calloc(roundsize, sizeof(fcs_float));
                    fcs_float * dy_array = calloc(roundsize, sizeof(fcs_float));
                    fcs_float * dz_array = calloc(roundsize, sizeof(fcs_float));

#pragma omp for schedule(static) private(j, pd_x, pd_y, pd_z, dx, dy, dz, ir, roundpos) reduction(+:p_sum, f_sum_zero, f_sum_one, f_sum_two) firstprivate(q1, xyz0, xyz1, box_a, box_b, box_c, cutoff, roundsize, pd_x_array, pd_y_array, pd_z_array)
                    for (j = 0; j < n1; ++j)
                    {
                        for (roundpos = 0; roundpos < roundsize; roundpos++)
                        {
                            dx_array[roundpos] = xyz0[i * 3 + 0] - xyz1[j * 3 + 0]
                                                 - (pd_x_array[roundpos] * box_a[0])
                                                 - (pd_y_array[roundpos] * box_b[0])
                                                 - (pd_z_array[roundpos] * box_c[0]);
                            dy_array[roundpos] = xyz0[i * 3 + 1] - xyz1[j * 3 + 1]
                                                 - (pd_x_array[roundpos] * box_a[1])
                                                 - (pd_y_array[roundpos] * box_b[1])
                                                 - (pd_z_array[roundpos] * box_c[1]);
                            dz_array[roundpos] = xyz0[i * 3 + 2] - xyz1[j * 3 + 2]
                                                 - (pd_x_array[roundpos] * box_a[2])
                                                 - (pd_y_array[roundpos] * box_b[2])
                                                 - (pd_z_array[roundpos] * box_c[2]);
                        }

                        for (roundpos = 0; roundpos < roundsize; roundpos++)
                        {
                            dx = dx_array[roundpos];
                            dy = dy_array[roundpos];
                            dz = dz_array[roundpos];

                            ir = 1.0 / fcs_sqrt(z_sqr(dx) + z_sqr(dy) + z_sqr(dz));

                            if ((cutoff > 0 && cutoff > ir) || (cutoff < 0 && -cutoff < ir))
                                continue;

                            fcs_float temptest = q1[j] * ir;
                            p_sum += temptest;

                            temptest *= ir * ir;
                            f_sum_zero += temptest * dx;
                            f_sum_one += temptest * dy;
                            f_sum_two += temptest * dz;
                        }

                    }

                    free(dx_array);
                    free(dy_array);
                    free(dz_array);
                }

#pragma omp critical
                {
                    p[i] += p_sum;

                    f[i * 3 + 0] += f_sum_zero;
                    f[i * 3 + 1] += f_sum_one;
                    f[i * 3 + 2] += f_sum_two;
                }
            }
            free(pd_x_array);
            free(pd_y_array);
            free(pd_z_array);
        }
