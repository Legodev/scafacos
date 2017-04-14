/*
  Copyright (C) 2011, 2012, 2013 Michael Hofmann
  
  This file is part of ScaFaCoS.
  
  ScaFaCoS is free software: you can redistribute it and/or modify
  it under the terms of the GNU Lesser Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  ScaFaCoS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser Public License for more details.
  
  You should have received a copy of the GNU Lesser Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <mpi.h>

#include "common/fcs-common/FCSCommon.h"

#include "common/gridsort/gridsort.h"
#include "common/near/near.h"

#include "z_tools.h"
#include "directc.h"

#define FCS_ENABLE_INTRINSIC
#ifdef FCS_ENABLE_INTRINSIC
#include <immintrin.h>
#include <malloc.h>
#endif


#if defined(FCS_ENABLE_DEBUG) || 0
# define DO_DEBUG
# define DEBUG_CMD(_cmd_)  Z_MOP(_cmd_)
#else
# define DEBUG_CMD(_cmd_)  Z_NOP()
#endif
#define DEBUG_PRINT_PREFIX  "DIRECT_DEBUG: "

#if defined(FCS_ENABLE_INFO) || 0
# define DO_INFO
# define INFO_CMD(_cmd_)  Z_MOP(_cmd_)
#else
# define INFO_CMD(_cmd_)  Z_NOP()
#endif
#define INFO_PRINT_PREFIX  "DIRECT_INFO: "

#if defined(FCS_ENABLE_TIMING) || 0
# define DO_TIMING
# define TIMING_CMD(_cmd_)  Z_MOP(_cmd_)
#else
# define TIMING_CMD(_cmd_)  Z_NOP()
#endif
#define TIMING_PRINT_PREFIX  "DIRECT_TIMING: "

#define MASTER_RANK  0

/*#define PRINT_PARTICLES*/

#define DO_TIMING_SYNC

#ifdef DO_TIMING
# define TIMING_DECL(_decl_)       _decl_
# define TIMING_CMD(_cmd_)         Z_MOP(_cmd_)
#else
# define TIMING_DECL(_decl_)
# define TIMING_CMD(_cmd_)         Z_NOP()
#endif
#ifdef DO_TIMING_SYNC
# define TIMING_SYNC(_c_)          TIMING_CMD(MPI_Barrier(_c_);)
#else
# define TIMING_SYNC(_c_)          Z_NOP()
#endif
#define TIMING_START(_t_)          TIMING_CMD(((_t_) = MPI_Wtime());)
#define TIMING_STOP(_t_)           TIMING_CMD(((_t_) = MPI_Wtime() - (_t_));)
#define TIMING_STOP_ADD(_t_, _r_)  TIMING_CMD(((_r_) += MPI_Wtime() - (_t_));)

#ifdef FCS_ENABLE_OFFLOADING
// defines for MIC Offloading
#define ALLOC alloc_if(1) free_if(0)
#define FREE alloc_if(0) free_if(1)
#define REUSE alloc_if(0) free_if(0)
// define attribut for offloadable functions
#define MICTARGETATTRIBUTE __attribute__((target(mic)))
#else
#define MICTARGETATTRIBUTE
#endif

void fcs_directc_create(fcs_directc_t *directc)
{
  directc->box_base[0] = directc->box_base[1] = directc->box_base[2] = 0;
  directc->box_a[0] = directc->box_a[1] = directc->box_a[2] = 0;
  directc->box_b[0] = directc->box_b[1] = directc->box_b[2] = 0;
  directc->box_c[0] = directc->box_c[1] = directc->box_c[2] = 0;
  directc->periodicity[0] = directc->periodicity[1] = directc->periodicity[2] = -1;

  directc->nparticles = directc->max_nparticles = 0;
  directc->positions = NULL;
  directc->charges = NULL;
  directc->field = NULL;
  directc->potentials = NULL;

  directc->in_nparticles = 0;
  directc->in_positions = NULL;
  directc->in_charges = NULL;

  directc->out_nparticles = 0;
  directc->out_positions = NULL;
  directc->out_field = NULL;
  directc->out_potentials = NULL;

  directc->periodic_images[0] = directc->periodic_images[1] = directc->periodic_images[2] = 1;
  directc->cutoff = 0.0;
  directc->cutoff_with_near = 0;

  directc->max_particle_move = -1;

  directc->resort = 0;
  directc->near_resort = FCS_NEAR_RESORT_NULL;
}


void fcs_directc_destroy(fcs_directc_t *directc)
{
  fcs_near_resort_destroy(&directc->near_resort);
}


void fcs_directc_set_system(fcs_directc_t *directc, const fcs_float *box_base, const fcs_float *box_a, const fcs_float *box_b, const fcs_float *box_c, const fcs_int *periodicity)
{
  fcs_int i;

  for (i = 0; i < 3; ++i)
  {
    directc->box_base[i] = box_base[i];
    directc->box_a[i] = box_a[i];
    directc->box_b[i] = box_b[i];
    directc->box_c[i] = box_c[i];

    if (periodicity) directc->periodicity[i] = periodicity[i];
  }
}


void fcs_directc_set_particles(fcs_directc_t *directc, fcs_int nparticles, fcs_int max_nparticles, fcs_float *positions, fcs_float *charges, fcs_float *field, fcs_float *potentials)
{
  directc->nparticles = nparticles;
  directc->max_nparticles = max_nparticles;
  directc->positions = positions;
  directc->charges = charges;
  directc->field = field;
  directc->potentials = potentials;
}


void fcs_directc_set_in_particles(fcs_directc_t *directc, fcs_int in_nparticles, fcs_float *in_positions, fcs_float *in_charges)
{
  directc->in_nparticles = in_nparticles;
  directc->in_positions = in_positions;
  directc->in_charges = in_charges;
}


void fcs_directc_set_out_particles(fcs_directc_t *directc, fcs_int out_nparticles, fcs_float *out_positions, fcs_float *out_field, fcs_float *out_potentials)
{
  directc->out_nparticles = out_nparticles;
  directc->out_positions = out_positions;
  directc->out_field = out_field;
  directc->out_potentials = out_potentials;
}


void fcs_directc_set_periodic_images(fcs_directc_t *directc, fcs_int *periodic_images)
{
  fcs_int i;

  for (i = 0; i < 3; ++i)
    directc->periodic_images[i] = periodic_images[i];
}


void fcs_directc_get_periodic_images(fcs_directc_t *directc, fcs_int *periodic_images)
{
  fcs_int i;

  for (i = 0; i < 3; ++i)
    periodic_images[i] = directc->periodic_images[i];
}


void fcs_directc_set_cutoff(fcs_directc_t *directc, fcs_float cutoff)
{
  directc->cutoff = cutoff;
}


void fcs_directc_get_cutoff(fcs_directc_t *directc, fcs_float *cutoff)
{
  *cutoff = directc->cutoff;
}


void fcs_directc_set_cutoff_with_near(fcs_directc_t *directc, fcs_int cutoff_with_near)
{
  directc->cutoff_with_near = cutoff_with_near;
}


void fcs_directc_get_cutoff_with_near(fcs_directc_t *directc, fcs_int *cutoff_with_near)
{
  *cutoff_with_near = directc->cutoff_with_near;
}


void fcs_directc_set_max_particle_move(fcs_directc_t *directc, fcs_float max_particle_move)
{
  directc->max_particle_move = max_particle_move;
}


void fcs_directc_set_resort(fcs_directc_t *directc, fcs_int resort)
{
  directc->resort = resort;
}


void fcs_directc_get_resort(fcs_directc_t *directc, fcs_int *resort)
{
  *resort = directc->resort;
}


void fcs_directc_get_resort_availability(fcs_directc_t *directc, fcs_int *availability)
{
  *availability = fcs_near_resort_is_available(directc->near_resort);
}


void fcs_directc_get_resort_particles(fcs_directc_t *directc, fcs_int *resort_particles)
{
  if (directc->near_resort == FCS_NEAR_RESORT_NULL)
  {
    *resort_particles = directc->nparticles;
    return;
  }
  
  *resort_particles = fcs_near_resort_get_sorted_particles(directc->near_resort);
}


#ifdef PRINT_PARTICLES
static void directc_print_particles(fcs_int n, fcs_float *xyz, fcs_float *q, fcs_float *f, fcs_float *p)
{
  fcs_int i;

  for (i = 0; i < n; ++i)
  {
    printf("  %" FCS_LMOD_INT "d: [%" FCS_LMOD_FLOAT "f,%" FCS_LMOD_FLOAT "f,%" FCS_LMOD_FLOAT "f] [%" FCS_LMOD_FLOAT "f] -> [%.2" FCS_LMOD_FLOAT "f,%.2" FCS_LMOD_FLOAT "f,%.2" FCS_LMOD_FLOAT "f] [%.2" FCS_LMOD_FLOAT "f]\n",
      i, xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2], q[i], f[i * 3 + 0], f[i * 3 + 1], f[i * 3 + 2], p[i]);
  }
}
#endif


static void
#ifdef FCS_ENABLE_OFFLOADING
#ifdef __INTEL_COMPILER
__attribute__((target(mic)))
#endif
#endif
directc_local_one(fcs_int nout, fcs_int nin, fcs_float *xyz, fcs_float *q, fcs_float *f, fcs_float *p, fcs_float cutoff)
{
  fcs_int i, j;
  fcs_float dx, dy, dz, ir;
  fcs_float p_sum, f_sum_zero, f_sum_one, f_sum_two;


  if (fcs_fabs(cutoff) > 0) cutoff = 1.0 / cutoff;

// not parallelizable because of access conflicts where inner loop accesses the field of the next outer loop
  for (i = 0; i < nout; ++i)
  {
    p_sum = 0.0;
    f_sum_zero = 0.0;
    f_sum_one = 0.0;
    f_sum_two = 0.0;

// parallelizable
#pragma omp parallel for schedule(static) private(j, dx, dy, dz, ir) firstprivate(i, q, xyz, cutoff) reduction(+:p_sum, f_sum_zero, f_sum_one, f_sum_two)
    for (j = i + 1; j < nout; ++j)
    {
      dx = xyz[i*3+0] - xyz[j*3+0];
      dy = xyz[i*3+1] - xyz[j*3+1];
      dz = xyz[i*3+2] - xyz[j*3+2];

      ir = 1.0 / fcs_sqrt(z_sqr(dx) + z_sqr(dy) + z_sqr(dz));

      if ((cutoff > 0 && cutoff > ir) || (cutoff < 0 && -cutoff < ir)) continue;

      p_sum += q[j] * ir;
      p[j] += q[i] * ir;

      f_sum_zero += q[j] * dx * ir * ir * ir;
      f_sum_one += q[j] * dy * ir * ir * ir;
      f_sum_two += q[j] * dz * ir * ir * ir;

      f[j*3+0] -= q[i] * dx * ir * ir * ir;
      f[j*3+1] -= q[i] * dy * ir * ir * ir;
      f[j*3+2] -= q[i] * dz * ir * ir * ir;
    }

#pragma omp parallel for schedule(static) private(j, dx, dy, dz, ir) firstprivate(i, q, xyz, cutoff) reduction(+:p_sum, f_sum_zero, f_sum_one, f_sum_two)
    for (j = nout; j < nin; ++j)
    {
      dx = xyz[i*3+0] - xyz[j*3+0];
      dy = xyz[i*3+1] - xyz[j*3+1];
      dz = xyz[i*3+2] - xyz[j*3+2];

      ir = 1.0 / fcs_sqrt(z_sqr(dx) + z_sqr(dy) + z_sqr(dz));

      if ((cutoff > 0 && cutoff > ir) || (cutoff < 0 && -cutoff < ir)) continue;

      p_sum += q[j] * ir;

      f_sum_zero += q[j] * dx * ir * ir * ir;
      f_sum_one += q[j] * dy * ir * ir * ir;
      f_sum_two += q[j] * dz * ir * ir * ir;
    }
#pragma omp critical
    {
      p[i] += p_sum;

      f[i*3+0] += f_sum_zero;
      f[i*3+1] += f_sum_one;
      f[i*3+2] += f_sum_two;
    }
  }
}


static void
#ifdef FCS_ENABLE_OFFLOADING
#ifdef __INTEL_COMPILER
__attribute__((target(mic)))
#endif
#endif
directc_local_two(fcs_int n0, fcs_float *xyz0, fcs_float *q0, fcs_int n1, fcs_float *xyz1, fcs_float *q1, fcs_float *f, fcs_float *p, fcs_float cutoff)
{
  fcs_int i, j;
  fcs_float dx, dy, dz, ir;
  fcs_float p_sum, f_sum_zero, f_sum_one, f_sum_two;


  if (fcs_fabs(cutoff) > 0) cutoff = 1.0 / cutoff;

#pragma omp parallel for schedule(static) private(i, j, dx, dy, dz, ir, p_sum, f_sum_zero, f_sum_one, f_sum_two) firstprivate(q1, xyz0, xyz1, cutoff) shared(p, f)
  for (i = 0; i < n0; ++i)
  {
    p_sum = 0.0;
    f_sum_zero = 0.0;
    f_sum_one = 0.0;
    f_sum_two = 0.0;

#pragma omp parallel for schedule(static) private(j, dx, dy, dz, ir) reduction(+:p_sum, f_sum_zero, f_sum_one, f_sum_two) firstprivate(xyz0, xyz1, cutoff)
    for (j = 0; j < n1; ++j)
    {
      dx = xyz0[i*3+0] - xyz1[j*3+0];
      dy = xyz0[i*3+1] - xyz1[j*3+1];
      dz = xyz0[i*3+2] - xyz1[j*3+2];

      ir = 1.0 / fcs_sqrt(z_sqr(dx) + z_sqr(dy) + z_sqr(dz));

      if ((cutoff > 0 && cutoff > ir) || (cutoff < 0 && -cutoff < ir)) continue;

      p_sum += q1[j] * ir;

      f_sum_zero += q1[j] * dx * ir * ir * ir;
      f_sum_one += q1[j] * dy * ir * ir * ir;
      f_sum_two += q1[j] * dz * ir * ir * ir;
    }

#pragma omp critical
    {
      p[i] += p_sum;

      f[i*3+0] += f_sum_zero;
      f[i*3+1] += f_sum_one;
      f[i*3+2] += f_sum_two;
    }
  }
}

static void
#ifdef FCS_ENABLE_OFFLOADING
#ifdef __INTEL_COMPILER
__attribute__((target(mic)))
#endif
#endif

#ifdef FCS_ENABLE_INTRINSIC
directc_local_periodic (fcs_int n0, fcs_float *xyz0, fcs_float *q0, fcs_int n1, fcs_float *xyz1, fcs_float *q1, fcs_float *f, fcs_float *p, fcs_int *periodic, fcs_float *box_a, fcs_float *box_b, fcs_float *box_c, fcs_float cutoff)
{
  fcs_int i, j, pd_x, pd_y, pd_z;
  fcs_float dx, dy, dz, ir;
  fcs_float p_sum, f_sum_zero, f_sum_one, f_sum_two;

  unsigned fcs_int roundsize = (2 * periodic[0] + 1) * (2 * periodic[1] + 1) * (2 * periodic[2] + 1) - 1;
  unsigned fcs_int roundpos = 0;
//  fcs_float * pd_x_array = (fcs_float *) _mm_malloc (roundsize * sizeof(fcs_int), 64);
//  fcs_float * pd_y_array = (fcs_float *) _mm_malloc (roundsize * sizeof(fcs_int), 64);
//  fcs_float * pd_z_array = (fcs_float *) _mm_malloc (roundsize * sizeof(fcs_int), 64);
  fcs_float pd_x_array[64] __attribute__((aligned(64)));
  fcs_float pd_y_array[64] __attribute__((aligned(64)));
  fcs_float pd_z_array[64] __attribute__((aligned(64)));

  for (pd_x = -periodic[0]; pd_x <= periodic[0]; ++pd_x)
    for (pd_y = -periodic[1]; pd_y <= periodic[1]; ++pd_y)
      for (pd_z = -periodic[2]; pd_z <= periodic[2]; ++pd_z)
	{
	  if (pd_x == 0 && pd_y == 0 && pd_z == 0)
	    continue;

	  pd_x_array[roundpos] = pd_x;
	  pd_y_array[roundpos] = pd_y;
	  pd_z_array[roundpos] = pd_z;

#ifdef PRINT_PARTICLES
#ifndef __MIC__
	  printf("%d - %d - %d - %d - %d\n", (int)pd_x_array[roundpos], (int)pd_y_array[roundpos], (int)pd_z_array[roundpos], roundpos, roundsize);
#endif
#endif

	  roundpos++;
	}

#ifdef __MIC__
// ignore very low cutoff to prevent floating point copy errors
  if (fcs_fabs(cutoff) < 0.000000001)
    {
#pragma omp parallel for schedule(static) private(i, j, pd_x, pd_y, pd_z, dx, dy, dz, ir, roundpos, p_sum, f_sum_zero, f_sum_one, f_sum_two) firstprivate(q1, xyz0, xyz1, box_a, box_b, box_c, cutoff, roundsize) shared(p, f, pd_x_array, pd_y_array, pd_z_array)
      for (i = 0; i < n0; ++i)
	{
	  p_sum = 0.0;
	  f_sum_zero = 0.0;
	  f_sum_one = 0.0;
	  f_sum_two = 0.0;

#pragma omp parallel num_threads(4)
	    {
//	      fcs_float * dx_array = (fcs_float *) _mm_malloc (roundsize * sizeof(fcs_float), 64);
//	      fcs_float * dy_array = (fcs_float *) _mm_malloc (roundsize * sizeof(fcs_float), 64);
//	      fcs_float * dz_array = (fcs_float *) _mm_malloc (roundsize * sizeof(fcs_float), 64);
	      fcs_float dx_array[64] __attribute__((aligned(64)));
	      fcs_float dy_array[64] __attribute__((aligned(64)));
	      fcs_float dz_array[64] __attribute__((aligned(64)));

#pragma omp for schedule(static) private(j, pd_x, pd_y, pd_z, dx, dy, dz, ir, roundpos) reduction(+:p_sum, f_sum_zero, f_sum_one, f_sum_two) firstprivate(q1, xyz0, xyz1, box_a, box_b, box_c, cutoff, roundsize, pd_x_array, pd_y_array, pd_z_array)
	      for (j = 0; j < n1; ++j)
		{
//		__m512d m512_xyz0_array = _mm512_set1_pd( xyz0[i * 3 + 0] - xyz1[j * 3 + 0]);
//		__m512d m512_xyz1_array = _mm512_set1_pd( xyz0[i * 3 + 1] - xyz1[j * 3 + 1]);
//		__m512d m512_xyz2_array = _mm512_set1_pd( xyz0[i * 3 + 2] - xyz1[j * 3 + 2]);

//		  __m512d m512_xyz00_array = _mm512_set1_pd (xyz0[i * 3 + 0]);
//		  __m512d m512_xyz01_array = _mm512_set1_pd (xyz0[i * 3 + 1]);
//		  __m512d m512_xyz02_array = _mm512_set1_pd (xyz0[i * 3 + 2]);
//
//		  __m512d m512_xyz10_array = _mm512_set1_pd (xyz1[j * 3 + 0]);
//		  __m512d m512_xyz11_array = _mm512_set1_pd (xyz1[j * 3 + 1]);
//		  __m512d m512_xyz12_array = _mm512_set1_pd (xyz1[j * 3 + 2]);
//
//		  __m512d m512_xyz0_array = _mm512_sub_pd (m512_xyz00_array, m512_xyz10_array);
//		  __m512d m512_xyz1_array = _mm512_sub_pd (m512_xyz01_array, m512_xyz11_array);
//		  __m512d m512_xyz2_array = _mm512_sub_pd (m512_xyz02_array, m512_xyz12_array);
//
//		  __m512d m512_box_a0_array = _mm512_set1_pd (box_a[0]);
//		  __m512d m512_box_a1_array = _mm512_set1_pd (box_a[1]);
//		  __m512d m512_box_a2_array = _mm512_set1_pd (box_a[2]);
//
//		  __m512d m512_box_b0_array = _mm512_set1_pd (box_b[0]);
//		  __m512d m512_box_b1_array = _mm512_set1_pd (box_b[1]);
//		  __m512d m512_box_b2_array = _mm512_set1_pd (box_b[2]);
//
//		  __m512d m512_box_c0_array = _mm512_set1_pd (box_c[0]);
//		  __m512d m512_box_c1_array = _mm512_set1_pd (box_c[1]);
//		  __m512d m512_box_c2_array = _mm512_set1_pd (box_c[2]);
//
//		  for (roundpos = 0; roundpos < roundsize; roundpos += 8)
//		    {
//		      __m512d m512_pd_dx_array = _mm512_load_pd (&pd_x_array[roundpos]);
//		      __m512d m512_pd_dy_array = _mm512_load_pd (&pd_y_array[roundpos]);
//		      __m512d m512_pd_dz_array = _mm512_load_pd (&pd_z_array[roundpos]);
//
//		      // dx_array
//		      __m512d m512_tmp = _mm512_mul_pd (m512_pd_dx_array, m512_box_a0_array);
//		      __m512d m512_dx_array = _mm512_sub_pd (m512_xyz0_array, m512_tmp);
//
//		      m512_tmp = _mm512_mul_pd (m512_pd_dy_array, m512_box_b0_array);
//		      m512_dx_array = _mm512_sub_pd (m512_dx_array, m512_tmp);
//
//		      m512_tmp = _mm512_mul_pd (m512_pd_dz_array, m512_box_c0_array);
//		      m512_dx_array = _mm512_sub_pd (m512_dx_array, m512_tmp);
//
//		      // dy_array
//		      m512_tmp = _mm512_mul_pd (m512_pd_dx_array, m512_box_a1_array);
//		      __m512d m512_dy_array = _mm512_sub_pd (m512_xyz1_array, m512_tmp);
//
//		      m512_tmp = _mm512_mul_pd (m512_pd_dy_array, m512_box_b1_array);
//		      m512_dy_array = _mm512_sub_pd (m512_dy_array, m512_tmp);
//
//		      m512_tmp = _mm512_mul_pd (m512_pd_dz_array, m512_box_c1_array);
//		      m512_dy_array = _mm512_sub_pd (m512_dy_array, m512_tmp);
//
//		      // dz_array
//		      m512_tmp = _mm512_mul_pd (m512_pd_dx_array, m512_box_a2_array);
//		      __m512d m512_dz_array = _mm512_sub_pd (m512_xyz2_array, m512_tmp);
//
//		      m512_tmp = _mm512_mul_pd (m512_pd_dy_array, m512_box_b2_array);
//		      m512_dz_array = _mm512_sub_pd (m512_dy_array, m512_tmp);
//
//		      m512_tmp = _mm512_mul_pd (m512_pd_dz_array, m512_box_c2_array);
//		      m512_dz_array = _mm512_sub_pd (m512_dy_array, m512_tmp);
//
//		      _mm512_store_pd (&dx_array[roundpos], m512_dx_array);
//		      _mm512_store_pd (&dy_array[roundpos], m512_dy_array);
//		      _mm512_store_pd (&dz_array[roundpos], m512_dz_array);
//		    }
//
//		  for (roundpos -= 8; roundpos < roundsize; roundpos++)
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

		      fcs_float temptest = q1[j] * ir;
		      p_sum += temptest;

		      temptest *= ir * ir;
		      f_sum_zero += temptest * dx;
		      f_sum_one += temptest * dy;
		      f_sum_two += temptest * dz;
		    }

		}

//	      _mm_free (dx_array);
//	      _mm_free (dy_array);
//	      _mm_free (dz_array);
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
  else
#endif
  {
#ifdef __MIC__
      printf ("the intrinsics code does not yet support cutoff, falling back to no intrinsics code. cutoff: %f\n", cutoff);
#endif
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
	      fcs_float * dx_array = calloc (roundsize, sizeof(fcs_float));
	      fcs_float * dy_array = calloc (roundsize, sizeof(fcs_float));
	      fcs_float * dz_array = calloc (roundsize, sizeof(fcs_float));

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

		      if ((cutoff > 0 && cutoff > ir)
			  || (cutoff < 0 && -cutoff < ir))
			continue;

		      fcs_float temptest = q1[j] * ir;
		      p_sum += temptest;

		      temptest *= ir * ir;
		      f_sum_zero += temptest * dx;
		      f_sum_one += temptest * dy;
		      f_sum_two += temptest * dz;
		    }

		}

	      free (dx_array);
	      free (dy_array);
	      free (dz_array);
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

  printf("\n\n STARTING FREE\n\n\n");
//  _mm_free (pd_x_array);
//  _mm_free (pd_y_array);
//  _mm_free (pd_z_array);
}
#else
directc_local_periodic (fcs_int n0, fcs_float *xyz0, fcs_float *q0, fcs_int n1, fcs_float *xyz1, fcs_float *q1, fcs_float *f, fcs_float *p, fcs_int *periodic, fcs_float *box_a, fcs_float *box_b, fcs_float *box_c, fcs_float cutoff)
{
  fcs_int i, j, pd_x, pd_y, pd_z;
  fcs_float dx, dy, dz, ir;
  fcs_float p_sum, f_sum_zero, f_sum_one, f_sum_two;

  if (fcs_fabs(cutoff) > 0)
    cutoff = 1.0 / cutoff;

#pragma omp parallel for schedule(static) private(i, j, pd_x, pd_y, pd_z, dx, dy, dz, ir, p_sum, f_sum_zero, f_sum_one, f_sum_two) firstprivate(q1, xyz0, xyz1, box_a, box_b, box_c, cutoff) shared(p, f)
  for (i = 0; i < n0; ++i)
  {
    p_sum = 0.0;
    f_sum_zero = 0.0;
    f_sum_one = 0.0;
    f_sum_two = 0.0;

#pragma omp parallel for schedule(static) private(j, pd_x, pd_y, pd_z, dx, dy, dz, ir) reduction(+:p_sum, f_sum_zero, f_sum_one, f_sum_two) firstprivate(q1, xyz0, xyz1, box_a, box_b, box_c, cutoff)
    for (j = 0; j < n1; ++j)
    for (pd_x = -periodic[0]; pd_x <= periodic[0]; ++pd_x)
    for (pd_y = -periodic[1]; pd_y <= periodic[1]; ++pd_y)
#pragma simd private(dx, dy, dz, ir) reduction(+:p_sum, f_sum_zero, f_sum_one, f_sum_two) firstprivate(pd_x, pd_y, j, q1, xyz0, xyz1, box_a, box_b, box_c, cutoff)
    for (pd_z = -periodic[2]; pd_z <= periodic[2]; ++pd_z)
    {
      if (pd_x == 0 && pd_y == 0 && pd_z == 0)
        continue;

      dx = xyz0[i * 3 + 0] - (xyz1[j * 3 + 0] + (pd_x * box_a[0]) + (pd_y * box_b[0]) + (pd_z * box_c[0]));
      dy = xyz0[i * 3 + 1] - (xyz1[j * 3 + 1] + (pd_x * box_a[1]) + (pd_y * box_b[1]) + (pd_z * box_c[1]));
      dz = xyz0[i * 3 + 2] - (xyz1[j * 3 + 2] + (pd_x * box_a[2]) + (pd_y * box_b[2]) + (pd_z * box_c[2]));

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
#endif


static void directc_global(fcs_directc_t *directc, fcs_int *periodic, int size, int rank, MPI_Comm comm)
{
  fcs_int l;

  fcs_int my_n, max_n, all_n[size], other_n, other_n_next;

  fcs_float *other_xyzq, *other_xyz, *other_q;

  MPI_Status status;

#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)

printf("\n\nDATATYPE: %s\n\n\n", STR(fcs_float));

  my_n = directc->nparticles + directc->in_nparticles;
  MPI_Allreduce(&my_n, &max_n, 1, FCS_MPI_INT, MPI_MAX, comm);
  MPI_Allgather(&my_n, 1, FCS_MPI_INT, all_n, 1, FCS_MPI_INT, comm);

  other_xyzq = calloc(max_n, 4*sizeof(fcs_float));

  other_n_next = all_n[rank];

  other_n = other_n_next;
  other_xyz = other_xyzq;
  other_q = other_xyzq + 3 * other_n;

  memcpy(other_xyz, directc->positions, directc->nparticles * 3 * sizeof(fcs_float));
  memcpy(other_q, directc->charges, directc->nparticles * sizeof(fcs_float));

  if (directc->in_nparticles > 0 && directc->in_positions && directc->in_charges)
  {
    memcpy(other_xyz + directc->nparticles * 3, directc->in_positions, directc->in_nparticles * 3 * sizeof(fcs_float));
    memcpy(other_q + directc->nparticles, directc->in_charges, directc->in_nparticles * sizeof(fcs_float));
  }

  fcs_int directc_nparticles = directc->nparticles;
  fcs_float *directc_positions = directc->positions;
  fcs_float *directc_charges = directc->charges;
  fcs_float *directc_field = directc->field;
  fcs_float *directc_potentials = directc->potentials;
  fcs_float *directc_box_a = directc->box_a;
  fcs_float *directc_box_b = directc->box_b;
  fcs_float *directc_box_c = directc->box_c;
  fcs_float directc_cutoff = directc->cutoff;
  
  /* directc_nparticles             unmodified      value
   * directc_positions              unmodified      array(directc_nparticles * 3)
   * directc_charges                    unused      array(directc_nparticles)
   * other_n                        unmodified      value
   * other_xyz                      unmodified      array(directc_nparticles * 3)
   * other_q                        unmodified      array(directc_nparticles)
   * directc_field                    modified      array(directc_nparticles * 3)
   * directc_potentials               modified      array(directc_nparticles * 3)
   * periodic                       unmodified      array(3)
   * directc_box_a                  unmodified      array(3)
   * directc_box_b                  unmodified      array(3)
   * directc_box_c                  unmodified      array(3)
   * directc_cutoff                   modified      value
   */
#ifdef FCS_ENABLE_OFFLOADING
#ifdef __INTEL_COMPILER
#pragma offload target(mic:0) in(directc_nparticles: ALLOC) \
                              in(directc_positions:length(directc_nparticles * 3) ALLOC) \
                              in(directc_charges:length(directc_nparticles) ALLOC) \
                              in(other_n:ALLOC) \
                              in(other_xyz:length(directc_nparticles * 3) ALLOC) \
                              in(other_q:length(directc_nparticles) ALLOC) \
                              in(periodic:length(3) ALLOC) \
                              in(directc_box_a:length(3) ALLOC) \
                              in(directc_box_b:length(3) ALLOC) \
                              in(directc_box_c:length(3) ALLOC) \
                              in(directc_field:length(directc_nparticles * 3) ALLOC) \
                              in(directc_potentials:length(directc_nparticles) ALLOC) \
                              in(directc_cutoff: ALLOC)
#endif
#endif
  {
  /* directc_nparticles         unmodified      value
   * other_n                    unmodified      value
   * other_xyz                  unmodified      array(directc_nparticles * 3)
   * other_q                    unmodified      array(directc_nparticles)
   * directc_field                modified      array(directc_nparticles * 3)
   * directc_potentials           modified      array(directc_nparticles)
   * directc_cutoff               modified      value
   */
    directc_local_one(directc_nparticles, other_n, other_xyz, other_q, directc_field, directc_potentials, directc_cutoff);

  /* directc_nparticles         unmodified      value
   * directc_positions          unmodified      array(directc_nparticles * 3)
   * directc_charges                unused      array(directc_nparticles)
   * other_n                    unmodified      value
   * other_xyz                  unmodified      array(directc_nparticles * 3)
   * other_q                    unmodified      array(directc_nparticles)
   * directc_field                modified      array(directc_nparticles * 3)
   * directc_potentials           modified      array(directc_nparticles * 3)
   * periodic                   unmodified      array(3)
   * directc_box_a              unmodified      array(3)
   * directc_box_b              unmodified      array(3)
   * directc_box_c              unmodified      array(3)
   * directc_cutoff               modified      value
   */
    directc_local_periodic(directc_nparticles, directc_positions, directc_charges, other_n, other_xyz, other_q, directc_field, directc_potentials, periodic, directc_box_a, directc_box_b, directc_box_c, directc_cutoff);
  }

  for (l = 1; l < size; ++l)
  {
    other_n_next = all_n[(rank - l + size) % size];
    
    MPI_Sendrecv_replace(other_xyzq, max_n * (3 + 1), FCS_MPI_FLOAT, (rank + 1) % size, 0, (rank - 1 + size) % size, 0, comm, &status);

    other_n = other_n_next;
    other_xyz = other_xyzq;
    other_q = other_xyzq + 3 * other_n;

#ifdef FCS_ENABLE_OFFLOADING
#ifdef __INTEL_COMPILER
#pragma offload target(mic:0) nocopy(directc_nparticles: REUSE) \
                              nocopy(directc_positions:length(directc_nparticles * 3) REUSE) \
                              nocopy(directc_charges:length(directc_nparticles) REUSE) \
                              in(other_n:REUSE) \
                              in(other_xyz:length(directc_nparticles * 3) REUSE) \
                              in(other_q:length(directc_nparticles) REUSE) \
                              nocopy(periodic:length(3) REUSE) \
                              nocopy(directc_box_a:length(3) REUSE) \
                              nocopy(directc_box_b:length(3) REUSE) \
                              nocopy(directc_box_c:length(3) REUSE) \
                              nocopy(directc_field:length(directc_nparticles * 3) REUSE) \
                              nocopy(directc_potentials:length(directc_nparticles) REUSE) \
                              nocopy(directc_cutoff: REUSE)
#endif
#endif
     {
      /* directc_nparticles             unmodified      value
       * directc_positions              unmodified      array(directc_nparticles * 3)
       * directc_charges                    unused      array(directc_nparticles)
       * other_n                        unmodified      value
       * other_xyz                      unmodified      array(directc_nparticles * 3)
       * other_q                        unmodified      array(directc_nparticles)
       * directc_field                    modified      array(directc_nparticles * 3)
       * directc_potentials               modified      array(directc_nparticles * 3)
       * directc_cutoff                   modified      value
       */
      directc_local_two(directc_nparticles, directc_positions, directc_charges, other_n, other_xyz, other_q, directc_field, directc_potentials, directc_cutoff);
      /* directc_nparticles             unmodified      value
       * directc_positions              unmodified      array(directc_nparticles * 3)
       * directc_charges                    unused      array(directc_nparticles)
       * other_n                        unmodified      value
       * other_xyz                      unmodified      array(directc_nparticles * 3)
       * other_q                        unmodified      array(directc_nparticles)
       * directc_field                    modified      array(directc_nparticles * 3)
       * directc_potentials               modified      array(directc_nparticles * 3)
       * periodic                       unmodified      array(3)
       * directc_box_a                  unmodified      array(3)
       * directc_box_b                  unmodified      array(3)
       * directc_box_c                  unmodified      array(3)
       * directc_cutoff                   modified      value
       */
      directc_local_periodic(directc_nparticles, directc_positions, directc_charges, other_n, other_xyz, other_q, directc_field, directc_potentials, periodic, directc_box_a, directc_box_b, directc_box_c, directc_cutoff);
    }
  }

#ifdef FCS_ENABLE_OFFLOADING
#ifdef __INTEL_COMPILER
#pragma offload_transfer target(mic:0) nocopy(directc_nparticles: FREE) \
                              nocopy(directc_positions:length(directc_nparticles * 3) FREE) \
                              nocopy(directc_charges:length(directc_nparticles) FREE) \
                              nocopy(other_n:FREE) \
                              nocopy(other_xyz:length(directc_nparticles * 3) FREE) \
                              nocopy(other_q:length(directc_nparticles) FREE) \
                              nocopy(periodic:length(3) FREE) \
                              nocopy(directc_box_a:length(3) FREE) \
                              nocopy(directc_box_b:length(3) FREE) \
                              nocopy(directc_box_c:length(3) FREE) \
                              out(directc_field:length(directc_nparticles * 3) FREE) \
                              out(directc_potentials:length(directc_nparticles) FREE) \
                              nocopy(directc_cutoff: FREE)
#endif
#endif

  free(other_xyzq);
}


static void directc_virial(fcs_int n, fcs_float *xyz, fcs_float *q, fcs_float *f, fcs_float *v, int size, int rank, MPI_Comm comm)
{
  fcs_int i;
  fcs_float my_v[9];


  for (i = 0; i < 9; ++i) my_v[i] = 0;

  for (i = 0; i < n; ++i)
  {
    my_v[0] += f[i*3+0] * q[i] * xyz[i*3+0];
    my_v[1] += f[i*3+0] * q[i] * xyz[i*3+1];
    my_v[2] += f[i*3+0] * q[i] * xyz[i*3+2];
    my_v[3] += f[i*3+1] * q[i] * xyz[i*3+0];
    my_v[4] += f[i*3+1] * q[i] * xyz[i*3+1];
    my_v[5] += f[i*3+1] * q[i] * xyz[i*3+2];
    my_v[6] += f[i*3+2] * q[i] * xyz[i*3+0];
    my_v[7] += f[i*3+2] * q[i] * xyz[i*3+1];
    my_v[8] += f[i*3+2] * q[i] * xyz[i*3+2];
  }

  MPI_Allreduce(my_v, v, 9, FCS_MPI_FLOAT, MPI_SUM, comm);
}


#define VSIZE(_v_)  fcs_sqrt(z_sqr((_v_)[0]) + z_sqr((_v_)[1]) + z_sqr((_v_)[2]))

static fcs_float get_periodic_factor(fcs_float *v0, fcs_float *v1, fcs_float *v2, fcs_float cutoff)
{
  fcs_float n[3], f;


  n[0] = v1[1] * v2[2] - v1[2] * v2[1];
  n[1] = v1[2] * v2[0] - v1[0] * v2[2];
  n[2] = v1[0] * v2[1] - v1[1] * v2[0];

  f = VSIZE(n) * cutoff / (n[0] * v0[0] + n[1] * v0[1] + n[2] * v0[2]);

  if (f < 0) f *= -1.0;

  return f;
}


static void directc_coulomb_field_potential(const void *param, fcs_float dist, fcs_float *f, fcs_float *p)
{
  *p = 1.0 / dist;
  *f = -(*p) * (*p);
}

static FCS_NEAR_LOOP_FP(directc_coulomb_loop_fp, directc_coulomb_field_potential)


void fcs_directc_run(fcs_directc_t *directc, MPI_Comm comm)
{
  fcs_int i;

  int comm_rank, comm_size;

  fcs_near_t near;
  fcs_int periodic[3] = { 0, 0, 0 };

#ifdef DO_TIMING
  double t;
#endif


  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  INFO_CMD(
    if (comm_rank == MASTER_RANK)
    {
      printf(INFO_PRINT_PREFIX "nparticles: %" FCS_LMOD_INT "d (max: %" FCS_LMOD_INT "d)\n", directc->nparticles, directc->max_nparticles);
      printf(INFO_PRINT_PREFIX "positions:  %p\n", directc->positions);
      printf(INFO_PRINT_PREFIX "charges:    %p\n", directc->charges);
      printf(INFO_PRINT_PREFIX "field:      %p\n", directc->field);
      printf(INFO_PRINT_PREFIX "potentials: %p\n", directc->potentials);
      printf(INFO_PRINT_PREFIX "periodicity: [%" FCS_LMOD_INT "d, %" FCS_LMOD_INT "d, %" FCS_LMOD_INT "d]\n", directc->periodicity[0], directc->periodicity[1], directc->periodicity[2]);
      printf(INFO_PRINT_PREFIX "box_base: [%" FCS_LMOD_FLOAT "f, %" FCS_LMOD_FLOAT "f, %" FCS_LMOD_FLOAT "f]\n", directc->box_base[0], directc->box_base[1], directc->box_base[2]);
      printf(INFO_PRINT_PREFIX "box_a: [%" FCS_LMOD_FLOAT "f, %" FCS_LMOD_FLOAT "f, %" FCS_LMOD_FLOAT "f]\n", directc->box_a[0], directc->box_a[1], directc->box_a[2]);
      printf(INFO_PRINT_PREFIX "box_b: [%" FCS_LMOD_FLOAT "f, %" FCS_LMOD_FLOAT "f, %" FCS_LMOD_FLOAT "f]\n", directc->box_b[0], directc->box_b[1], directc->box_b[2]);
      printf(INFO_PRINT_PREFIX "box_c: [%" FCS_LMOD_FLOAT "f, %" FCS_LMOD_FLOAT "f, %" FCS_LMOD_FLOAT "f]\n", directc->box_c[0], directc->box_c[1], directc->box_c[2]);
      printf(INFO_PRINT_PREFIX "cutoff: %" FCS_LMOD_FLOAT "f, with near: %" FCS_LMOD_INT "d\n", directc->cutoff, directc->cutoff_with_near);
    }
  );

#ifdef FCS_ENABLE_OFFLOADING
// initialize mic offloading
#pragma offload_transfer target(mic)
#endif

  periodic[0] = directc->periodicity[0] * directc->periodic_images[0];
  periodic[1] = directc->periodicity[1] * directc->periodic_images[1];
  periodic[2] = directc->periodicity[2] * directc->periodic_images[2];

  if (directc->cutoff > 0.0)
  {
    if (directc->periodicity[0]) periodic[0] = (fcs_int) fcs_ceil(get_periodic_factor(directc->box_a, directc->box_b, directc->box_c, directc->cutoff));
    if (directc->periodicity[1]) periodic[1] = (fcs_int) fcs_ceil(get_periodic_factor(directc->box_b, directc->box_c, directc->box_a, directc->cutoff));
    if (directc->periodicity[2]) periodic[2] = (fcs_int) fcs_ceil(get_periodic_factor(directc->box_c, directc->box_a, directc->box_b, directc->cutoff));
  }

  INFO_CMD(
    if (comm_rank == MASTER_RANK) printf(INFO_PRINT_PREFIX "periodic: %" FCS_LMOD_INT "d / %" FCS_LMOD_INT "d / %" FCS_LMOD_INT "d\n", periodic[0], periodic[1], periodic[2]);
  );

  for (i = 0; i < directc->nparticles; ++i) directc->field[i * 3 + 0] = directc->field[i * 3 + 1] = directc->field[i * 3 + 2] = directc->potentials[i] = 0.0;

#ifdef PRINT_PARTICLES
  printf("%d:   particles IN:\n", comm_rank);
  directc_print_particles(directc->nparticles, directc->positions, directc->charges, directc->field, directc->potentials);
#endif

  TIMING_SYNC(comm); TIMING_START(t);

  if (directc->cutoff_with_near)
  {
    fcs_near_create(&near);

    fcs_near_set_loop(&near, directc_coulomb_loop_fp);
    fcs_near_set_system(&near, directc->box_base, directc->box_a, directc->box_b, directc->box_c, periodic);
    fcs_near_set_particles(&near, directc->nparticles, directc->max_nparticles, directc->positions, directc->charges, NULL, directc->field, directc->potentials);
    fcs_near_set_max_particle_move(&near, directc->max_particle_move);
    fcs_near_set_resort(&near, directc->resort);

    fcs_near_field_solver(&near, fabs(directc->cutoff), NULL, comm);

    if (directc->resort)
    {
      fcs_near_resort_destroy(&directc->near_resort);

      fcs_near_resort_create(&directc->near_resort, &near);
/*      fcs_near_resort_print(directc->near_resort, comm);*/
    }

    fcs_near_destroy(&near);

  } else
  {
    directc_global(directc, periodic, comm_size, comm_rank, comm);
  }

  TIMING_SYNC(comm); TIMING_STOP(t);

  directc_virial(directc->nparticles, directc->positions, directc->charges, directc->field, directc->virial, comm_size, comm_rank, comm);

#ifdef PRINT_PARTICLES
  printf("%d:   particles OUT:\n", comm_rank);
  directc_print_particles(directc->nparticles, directc->positions, directc->charges, directc->field, directc->potentials);
#endif

  TIMING_CMD(
    if (comm_rank == MASTER_RANK)
      printf(TIMING_PRINT_PREFIX "directc: %f\n", t);
  );
}


void fcs_directc_resort_ints(fcs_directc_t *directc, fcs_int *src, fcs_int *dst, fcs_int n, MPI_Comm comm)
{
  if (directc->near_resort == FCS_NEAR_RESORT_NULL) return;
  
  fcs_near_resort_ints(directc->near_resort, src, dst, n, comm);
}


void fcs_directc_resort_floats(fcs_directc_t *directc, fcs_float *src, fcs_float *dst, fcs_int n, MPI_Comm comm)
{
  if (directc->near_resort == FCS_NEAR_RESORT_NULL) return;
  
  fcs_near_resort_floats(directc->near_resort, src, dst, n, comm);
}


void fcs_directc_resort_bytes(fcs_directc_t *directc, void *src, void *dst, fcs_int n, MPI_Comm comm)
{
  if (directc->near_resort == FCS_NEAR_RESORT_NULL) return;
  
  fcs_near_resort_bytes(directc->near_resort, src, dst, n, comm);
}
