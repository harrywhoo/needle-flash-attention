#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid >= size) return;

  size_t num_dims = shape.size;
  size_t compact_idx = gid;
  size_t strided_pos = offset;

  for (int d = num_dims - 1; d >= 0; d--) {
    size_t coord = compact_idx % shape.data[d];
    compact_idx /= shape.data[d];
    strided_pos += coord * strides.data[d];
  }

  out[gid] = a[strided_pos];
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}



__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid >= size) return;

  size_t num_dims = shape.size;
  size_t compact_idx = gid;
  size_t strided_pos = offset;

  for (int d = num_dims - 1; d >= 0; d--) {
    size_t coord = compact_idx % shape.data[d];
    compact_idx /= shape.data[d];
    strided_pos += coord * strides.data[d];
  }

  out[strided_pos] = a[gid];
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape), VecToCuda(strides), offset);
  /// END SOLUTION
}



__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                                    CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid >= size) return;

  size_t num_dims = shape.size;
  size_t compact_idx = gid;
  size_t strided_pos = offset;

  for (int d = num_dims - 1; d >= 0; d--) {
    size_t coord = compact_idx % shape.data[d];
    compact_idx /= shape.data[d];
    strided_pos += coord * strides.data[d];
  }

  out[strided_pos] = val;
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape), VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN SOLUTION

#define EWISE_BINARY_OP(name, op) \
__global__ void Ewise##name##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = a[gid] op b[gid]; \
} \
void Ewise##name(const CudaArray& a, const CudaArray& b, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
}

#define SCALAR_BINARY_OP(name, op) \
__global__ void Scalar##name##Kernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = a[gid] op val; \
} \
void Scalar##name(const CudaArray& a, scalar_t val, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Scalar##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
}

#define EWISE_UNARY_OP(name, func) \
__global__ void Ewise##name##Kernel(const scalar_t* a, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = func(a[gid]); \
} \
void Ewise##name(const CudaArray& a, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size); \
}

EWISE_BINARY_OP(Mul, *)
SCALAR_BINARY_OP(Mul, *)

EWISE_BINARY_OP(Div, /)
SCALAR_BINARY_OP(Div, /)

EWISE_BINARY_OP(Eq, ==)
SCALAR_BINARY_OP(Eq, ==)

EWISE_BINARY_OP(Ge, >=)
SCALAR_BINARY_OP(Ge, >=)

__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = pow(a[gid], val);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = max(a[gid], b[gid]);
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = max(a[gid], val);
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

EWISE_UNARY_OP(Log, log)
EWISE_UNARY_OP(Exp, exp)
EWISE_UNARY_OP(Tanh, tanh)

/// END SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void MatmulKernel(const scalar_t* A, const scalar_t* B, scalar_t* C,
                             uint32_t M, uint32_t N, uint32_t P) {
  __shared__ scalar_t sA[TILE][TILE];
  __shared__ scalar_t sB[TILE][TILE];

  uint32_t ybase = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t xbase = blockIdx.x * blockDim.x + threadIdx.x;

  scalar_t c = 0.0f;

  // Tile over dimension N
  for (uint32_t ko = 0; ko < N; ko += TILE) {
    // Cooperative fetch
    // load tile from A (M x N)
    uint32_t a_col = ko + threadIdx.x;
    if (ybase < M && a_col < N) {
      sA[threadIdx.y][threadIdx.x] = A[ybase*N + a_col];
    } else {
      sA[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Cooperative fetch
    // load tile from B (N x P)
    uint32_t b_row = ko + threadIdx.y;
    if (b_row < N && xbase < P) {
      sB[threadIdx.y][threadIdx.x] = B[b_row*P + xbase];
    } else {
      sB[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    // Compute using shared memory
    for (uint32_t ki = 0; ki < TILE; ki++) {
      c += sA[threadIdx.y][ki] * sB[ki][threadIdx.x];
    }
    __syncthreads();
  }
  // Write result
  if (ybase < M && xbase < P) {
    C[ybase * P + xbase] = c;
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  dim3 block(TILE, TILE);
  dim3 grid((P + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}



////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t out_size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < out_size) {
    size_t start = gid * reduce_size;
    scalar_t max_val = a[start];
    for (size_t i = 1; i < reduce_size; i++) {
      max_val = max(max_val, a[start + i]);
    }
    out[gid] = max_val;
  }
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t out_size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < out_size) {
    size_t start = gid * reduce_size;
    scalar_t sum = 0.0f;
    for (size_t i = 0; i < reduce_size; i++) {
      sum += a[start + i];
    }
    out[gid] = sum;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}



void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you
   * can perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// FlashAttention operations
////////////////////////////////////////////////////////////////////////////////

__global__ void FlashAttentionForwardKernel(
    const scalar_t* __restrict__ Q, 
    const scalar_t* __restrict__ K, 
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ O, 
    scalar_t* __restrict__ m, 
    scalar_t* __restrict__ l,
    uint32_t seq_len, uint32_t head_dim,
    uint32_t block_m, uint32_t block_n, bool causal) 
{
    int tx = threadIdx.x;
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int offset_qkv = (batch_idx * gridDim.y * seq_len * head_dim) + (head_idx * seq_len * head_dim);
    int offset_ml  = (batch_idx * gridDim.y * seq_len) + (head_idx * seq_len);

    const scalar_t* q_ptr = Q + offset_qkv;
    const scalar_t* k_ptr = K + offset_qkv;
    const scalar_t* v_ptr = V + offset_qkv;
    scalar_t* o_ptr = O + offset_qkv;
    scalar_t* m_ptr = m + offset_ml;
    scalar_t* l_ptr = l + offset_ml;

    for (int i_base = 0; i_base < seq_len; i_base += block_m) {
        int i_end = min(i_base + block_m, seq_len);
        for (int i = i_base; i < i_end; ++i) {
            scalar_t m_i = -INFINITY;
            scalar_t l_i = 0.0f;
            scalar_t acc_o = 0.0f; 

            for (int j_base = 0; j_base < seq_len; j_base += block_n) {
                int j_end = min(j_base + block_n, seq_len);
                for (int j = j_base; j < j_end; ++j) {
                    if (causal && j > i) continue;

                    scalar_t score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        score += q_ptr[i * head_dim + d] * k_ptr[j * head_dim + d];
                    }
                    score /= sqrtf((float)head_dim);

                    scalar_t m_prev = m_i;
                    m_i = max(m_prev, score);
                    scalar_t P_ij = expf(score - m_i);
                    
                    l_i = expf(m_prev - m_i) * l_i + P_ij;
                    
                    if (tx < head_dim) {
                         acc_o = acc_o * expf(m_prev - m_i) + P_ij * v_ptr[j * head_dim + tx];
                    }
                }
            }

            if (tx < head_dim) {
                o_ptr[i * head_dim + tx] = acc_o / l_i;
            }
            
            if (tx == 0) {
                m_ptr[i] = m_i;
                l_ptr[i] = l_i;
            }
        }
    }
}


// ============================================================================
// BACKWARD KERNELS
// ============================================================================

__global__ void ComputeDeltaKernel(
    const scalar_t* __restrict__ dO,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ m,
    const scalar_t* __restrict__ l,
    scalar_t* __restrict__ delta,
    uint32_t seq_len, uint32_t head_dim, bool causal
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len) return;

    int batch_idx = blockIdx.y;
    int head_idx = blockIdx.z;

    int offset_qkv = (batch_idx * gridDim.z * seq_len * head_dim) + (head_idx * seq_len * head_dim);
    int offset_ml  = (batch_idx * gridDim.z * seq_len) + (head_idx * seq_len);

    const scalar_t* q_ptr = Q + offset_qkv;
    const scalar_t* k_ptr = K + offset_qkv;
    const scalar_t* v_ptr = V + offset_qkv;
    const scalar_t* do_ptr = dO + offset_qkv;
    const scalar_t* m_ptr = m + offset_ml;
    const scalar_t* l_ptr = l + offset_ml;
    scalar_t* delta_ptr = delta + offset_ml;

    scalar_t m_i = m_ptr[i];
    scalar_t l_i = l_ptr[i];
    scalar_t d_i = 0.0f;

    for (int j = 0; j < seq_len; ++j) {
        if (causal && j > i) continue;

        scalar_t score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_ptr[i * head_dim + d] * k_ptr[j * head_dim + d];
        }
        score /= sqrtf((float)head_dim);
        scalar_t p_ij = expf(score - m_i) / l_i;

        scalar_t do_dot_v = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            do_dot_v += do_ptr[i * head_dim + d] * v_ptr[j * head_dim + d];
        }

        d_i += p_ij * do_dot_v;
    }
    
    delta_ptr[i] = d_i;
}

__global__ void FlashAttentionBackwardKernel(
    const scalar_t* __restrict__ dO,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ m,
    const scalar_t* __restrict__ l,
    const scalar_t* __restrict__ delta,
    scalar_t* __restrict__ dQ,
    scalar_t* __restrict__ dK,
    scalar_t* __restrict__ dV,
    uint32_t seq_len, uint32_t head_dim, bool causal
) {
    int tx = threadIdx.x; 
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    int offset_qkv = (batch_idx * gridDim.y * seq_len * head_dim) + (head_idx * seq_len * head_dim);
    int offset_ml  = (batch_idx * gridDim.y * seq_len) + (head_idx * seq_len);

    const scalar_t* q_ptr = Q + offset_qkv;
    const scalar_t* k_ptr = K + offset_qkv;
    const scalar_t* v_ptr = V + offset_qkv;
    const scalar_t* do_ptr = dO + offset_qkv;
    
    scalar_t* dq_ptr = dQ + offset_qkv;
    scalar_t* dk_ptr = dK + offset_qkv;
    scalar_t* dv_ptr = dV + offset_qkv;
    
    const scalar_t* m_ptr = m + offset_ml;
    const scalar_t* l_ptr = l + offset_ml;
    const scalar_t* delta_ptr = delta + offset_ml;
    
    // Scaling factor for Q and K gradients
    scalar_t scale = 1.0f / sqrtf((float)head_dim);

    for (int i = 0; i < seq_len; ++i) {
        scalar_t m_i = m_ptr[i];
        scalar_t l_i = l_ptr[i];
        scalar_t d_i = delta_ptr[i];

        scalar_t acc_dq = 0.0f; 

        for (int j = 0; j < seq_len; ++j) {
            if (causal && j > i) continue;

            scalar_t score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_ptr[i * head_dim + d] * k_ptr[j * head_dim + d];
            }
            score /= sqrtf((float)head_dim);
            scalar_t p_ij = expf(score - m_i) / l_i;

            if (tx < head_dim) {
                scalar_t val = p_ij * do_ptr[i * head_dim + tx];
                atomicAdd(&dv_ptr[j * head_dim + tx], val);
            }

            scalar_t do_dot_v = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                do_dot_v += do_ptr[i * head_dim + d] * v_ptr[j * head_dim + d];
            }
            scalar_t ds_ij = p_ij * (do_dot_v - d_i);

            // 3. Update dQ with Scale
            if (tx < head_dim) {
                acc_dq += ds_ij * k_ptr[j * head_dim + tx] * scale;
            }

            // 4. Update dK with Scale
            if (tx < head_dim) {
                 scalar_t val = ds_ij * q_ptr[i * head_dim + tx] * scale;
                 atomicAdd(&dk_ptr[j * head_dim + tx], val);
            }
        }
        
        if (tx < head_dim) {
            dq_ptr[i * head_dim + tx] = acc_dq;
        }
    }
}


void FlashAttentionForward(
    const CudaArray& Q, const CudaArray& K, const CudaArray& V,
    CudaArray* O, CudaArray* m, CudaArray* l,
    uint32_t batch_size, uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
    uint32_t block_m, uint32_t block_n, bool causal) 
{
    dim3 grid(batch_size, num_heads);
    dim3 block(head_dim); 

    FlashAttentionForwardKernel<<<grid, block>>>(
        Q.ptr, K.ptr, V.ptr,
        O->ptr, m->ptr, l->ptr,
        seq_len, head_dim, block_m, block_n, causal
    );
}

void FlashAttentionBackward(
    const CudaArray& dO, const CudaArray& Q, const CudaArray& K, const CudaArray& V,
    const CudaArray& m, const CudaArray& l,
    CudaArray* dQ, CudaArray* dK, CudaArray* dV,
    uint32_t batch_size, uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
    uint32_t block_m, uint32_t block_n, bool causal) 
{
    size_t delta_size = batch_size * num_heads * seq_len * sizeof(scalar_t);
    scalar_t* d_ptr;
    cudaMalloc(&d_ptr, delta_size);

    size_t grad_size = batch_size * num_heads * seq_len * head_dim * sizeof(scalar_t);
    cudaMemset(dQ->ptr, 0, grad_size);
    cudaMemset(dK->ptr, 0, grad_size);
    cudaMemset(dV->ptr, 0, grad_size);

    dim3 delta_grid((seq_len + 31) / 32, batch_size, num_heads);
    dim3 delta_block(32);
    ComputeDeltaKernel<<<delta_grid, delta_block>>>(
        dO.ptr, Q.ptr, K.ptr, V.ptr, m.ptr, l.ptr, d_ptr,
        seq_len, head_dim, causal
    );
    
    dim3 grid(batch_size, num_heads);
    dim3 block(head_dim); 
    FlashAttentionBackwardKernel<<<grid, block>>>(
        dO.ptr, Q.ptr, K.ptr, V.ptr, m.ptr, l.ptr, d_ptr,
        dQ->ptr, dK->ptr, dV->ptr,
        seq_len, head_dim, causal
    );

    cudaFree(d_ptr);
}


}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);

  // FlashAttention operations
  m.def("flash_attention_forward", FlashAttentionForward);
  m.def("flash_attention_backward", FlashAttentionBackward);
}
