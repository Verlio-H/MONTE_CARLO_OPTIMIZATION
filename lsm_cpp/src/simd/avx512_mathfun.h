// Adapted from avx_mathfun.h
#include <immintrin.h>

#define exp_hi (88.3762626647949f)
#define exp_lo (-88.3762626647949f)

#define cephes_LOG2EF (1.44269504088896341)
#define cephes_exp_C1 (0.693359375)
#define cephes_exp_C2 (-2.12194440e-4)

#define cephes_exp_p0 (1.9875691500E-4)
#define cephes_exp_p1 (1.3981999507E-3)
#define cephes_exp_p2 (8.3334519073E-3)
#define cephes_exp_p3 (4.1665795894E-2)
#define cephes_exp_p4 (1.6666665459E-1)
#define cephes_exp_p5 (5.0000001201E-1)

__m512 exp_ps(__m512 x) {
  __m512 tmp = _mm512_setzero_ps(), fx;
  __m512i imm0;
  __m512 one = _mm512_set1_ps(1);

  x = _mm512_min_ps(x, _mm512_set1_ps(exp_hi));
  x = _mm512_max_ps(x, _mm512_set1_ps(exp_lo));

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm512_mul_ps(x, _mm512_set1_ps(cephes_LOG2EF));
  fx = _mm512_add_ps(fx, _mm512_set1_ps(0.5f));

  /* how to perform a floorf with SSE: just below */
  //imm0 = _mm256_cvttps_epi32(fx);
  //tmp  = _mm256_cvtepi32_ps(imm0);
  
  tmp = _mm512_floor_ps(fx);

  /* if greater, substract 1 */
  //v8sf mask = _mm256_cmpgt_ps(tmp, fx);    
  __mmask16 mask = _mm512_cmp_ps_mask(tmp, fx, _CMP_GT_OS);    
  fx = _mm512_mask_sub_ps(tmp, mask, tmp, one);

  tmp = _mm512_mul_ps(fx, _mm512_set1_ps(cephes_exp_C1));
  __m512 z = _mm512_mul_ps(fx, _mm512_set1_ps(cephes_exp_C2));
  x = _mm512_sub_ps(x, tmp);
  x = _mm512_sub_ps(x, z);

  z = _mm512_mul_ps(x,x);
  
  __m512 y = _mm512_set1_ps(cephes_exp_p0);
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, _mm512_set1_ps(cephes_exp_p1));
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, _mm512_set1_ps(cephes_exp_p2));
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, _mm512_set1_ps(cephes_exp_p3));
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, _mm512_set1_ps(cephes_exp_p4));
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, _mm512_set1_ps(cephes_exp_p5));
  y = _mm512_mul_ps(y, z);
  y = _mm512_add_ps(y, x);
  y = _mm512_add_ps(y, one);

  /* build 2^n */
  imm0 = _mm512_cvttps_epi32(fx);
  // another two AVX2 instructions
  imm0 = _mm512_add_epi32(imm0, _mm512_set1_epi32(0x7f));
  imm0 = _mm512_slli_epi32(imm0, 23);
  __m512 pow2n = _mm512_castsi512_ps(imm0);
  y = _mm512_mul_ps(y, pow2n);
  return y;
}

