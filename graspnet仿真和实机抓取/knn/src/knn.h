/*#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#endif



int knn(torch::Tensor& ref, torch::Tensor& query, torch::Tensor& idx)
{

    // TODO check dimensions
    int batch, ref_nb, query_nb, dim, k;
    batch = ref.size(0);
    dim = ref.size(1);
    k = idx.size(1);
    ref_nb = ref.size(2);
    query_nb = query.size(2);

    float *ref_dev = ref.data<float>();
    float *query_dev = query.data<float>();
    int *idx_dev = idx.data<int>();




  if (ref.type().is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    float *dist_dev = (float*)c10::cuda::CUDACachingAllocator::raw_alloc(ref_nb * query_nb * sizeof(float));
    for (int b = 0; b < batch; b++)
    {
      knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
      dist_dev, idx_dev + b * k * query_nb, c10::cuda::getCurrentCUDAStream());
    }
    c10::cuda::CUDACachingAllocator::raw_delete(dist_dev);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in knn: %s\n", cudaGetErrorString(err));
        // THError("aborting");
    }
    return 1;

    AT_ERROR("Not compiled with GPU support");

  }


    float *dist_dev = (float*)malloc(ref_nb * query_nb * sizeof(float));
    int *ind_buf = (int*)malloc(ref_nb * sizeof(int));
    for (int b = 0; b < batch; b++) {
    knn_cpu(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
      dist_dev, idx_dev + b * k * query_nb, ind_buf);
    }

    free(dist_dev);
    free(ind_buf);

    return 1;

}
*/

#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#include <torch/extension.h>
#endif

int knn(torch::Tensor &ref, torch::Tensor &query, torch::Tensor &idx)
{
  // TODO check dimensions
  int batch, ref_nb, query_nb, dim, k;
  batch = ref.size(0);
  dim = ref.size(1);
  k = idx.size(1);
  ref_nb = ref.size(2);
  query_nb = query.size(2);

  float *ref_dev = ref.data_ptr<float>();
  float *query_dev = query.data_ptr<float>();
  int *idx_dev = idx.data_ptr<int>();

  if (ref.is_cuda())
  {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    auto dist_dev = torch::empty({ref_nb * query_nb}, ref.options().device(torch::kCUDA));
    float *dist_dev_ptr = dist_dev.data_ptr<float>();

    for (int b = 0; b < batch; b++)
    {
      knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
                 dist_dev_ptr, idx_dev + b * k * query_nb, c10::cuda::getCurrentCUDAStream());
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      printf("error in knn: %s\n", cudaGetErrorString(err));
      AT_ERROR("aborting");
    }
    return 1;
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  auto dist_dev = torch::empty({ref_nb * query_nb}, ref.options().device(torch::kCPU));
  float *dist_dev_ptr = dist_dev.data_ptr<float>();
  auto ind_buf = torch::empty({ref_nb}, ref.options().dtype(torch::kInt32).device(torch::kCPU));
  int *ind_buf_ptr = ind_buf.data_ptr<int>();

  for (int b = 0; b < batch; b++)
  {
    knn_cpu(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
            dist_dev_ptr, idx_dev + b * k * query_nb, ind_buf_ptr);
  }

  return 1;
}
