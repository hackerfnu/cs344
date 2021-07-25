/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

__device__ float _min(float a, float b) {
	return a < b ? a : b;
}

__device__ float _max(float a, float b) {
	return a > b ? a : b;
}

__global__ void shmem_reduce(float *d_extreme, const float *d_in, bool isMin, int input_size) 
{
   // Initializing Shared Data in the block
   extern __shared__ float sdata[];
   int globalId = blockDim.x * blockIdx.x + threadIdx.x;
   uint tid = threadIdx.x;

   // Loading data from global -> shared(block mem)
   // Uses tid instead of globalId 
   // because tid is threadId within the block
   if (globalId >= input_size) { sdata[tid] = d_in[0]; } //dummy init (does not modify the final result)
   else sdata[tid] = d_in[globalId];
   __syncthreads(); //Barrier to wait for all threads to load data

   for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
      if(tid < s) {
         sdata[tid] = isMin ? _min(sdata[tid],sdata[tid + s]) : _max(sdata[tid],sdata[tid + s]);
      }
      __syncthreads(); //Barrier to wait for all threads to do operation
   }
   //Only thread 0 writes back result from this block
   if (tid == 0) {
      d_extreme[blockIdx.x] = sdata[tid];
   }
}

void reduce(const float* d_logLuminance,float &min_logLum,float &max_logLum,const int input_size){
   // Max number of Threads/Block is typically 1024
   // Parallel Reduction need to be size of 2**n
   const int kBlockSize = 1024;
   float *d_current_min = NULL;
   float *d_current_max = NULL;
   int size = input_size;
   int sh_mem = kBlockSize*sizeof(float);
   // Iterate until only one grid left and array reduced to one number
   while(true) {
      int grid_size = (kBlockSize + size - 1)/kBlockSize;
      float *d_min, *d_max;
      hipMalloc(&d_min, grid_size*sizeof(float));
      hipMalloc(&d_max, grid_size*sizeof(float));
      if(d_current_min == NULL || d_current_max == NULL) {
         // Min reduction
         shmem_reduce<<<grid_size, kBlockSize, sh_mem>>>(d_min, d_logLuminance, true, size);
         // Max reduction
         shmem_reduce<<<grid_size, kBlockSize, sh_mem>>>(d_max, d_logLuminance, false, size);
      } else {
         // Min reduction
         shmem_reduce<<<grid_size, kBlockSize, sh_mem>>>(d_min, d_current_min, true, size);
         // Max reduction
         shmem_reduce<<<grid_size, kBlockSize, sh_mem>>>(d_max, d_current_max, false, size);
      }
      hipDeviceSynchronize(); checkHIPErrors(hipGetLastError());
		if (d_current_min != NULL) checkHIPErrors(hipFree(d_current_min));
		if (d_current_max != NULL) checkHIPErrors(hipFree(d_current_max));

      if (grid_size == 1) {
			//end of reduction reached
			checkHIPErrors(hipMemcpy(&min_logLum, d_min, sizeof(float), hipMemcpyDeviceToHost));
			checkHIPErrors(hipMemcpy(&max_logLum, d_max, sizeof(float), hipMemcpyDeviceToHost));
         hipDeviceSynchronize(); checkHIPErrors(hipGetLastError());
         return;
		}

      size = grid_size; //Set new size of array ~ current size of grid
      if (grid_size == 0) grid_size++;
		d_current_min = d_min; //point to new intermediate result
		d_current_max = d_max; //point to new intermediate result
   }
}

__global__ void scatter_add(uint * d_out, const float *d_lum, const float lumMin, const float lumRange, const int numBins, const size_t inputSize) {
   int globalId = blockDim.x * blockIdx.x + threadIdx.x;
   if(globalId > inputSize) return;
   int binId = (d_lum[globalId] - lumMin) / lumRange * numBins;
   binId = binId == numBins ? numBins - 1 : binId; //max value bin is the last of the histo
   atomicAdd(&(d_out[binId]),1);
}

unsigned int* compute_histogram(const float* const d_logLuminance, int numBins, int input_size, float minVal, float rangeVals) {
	unsigned int* d_histo;
	checkHIPErrors(hipMalloc(&d_histo, numBins * sizeof(unsigned int)));
	checkHIPErrors(hipMemset(d_histo, 0, numBins * sizeof(unsigned int)));
	const int kBlockSize = 1024;
   const int kGridSize = (kBlockSize + input_size - 1)/kBlockSize;
	scatter_add<<<kGridSize, kBlockSize>>>(d_histo, d_logLuminance, minVal, rangeVals, numBins, input_size);
	hipDeviceSynchronize(); checkHIPErrors(hipGetLastError());
	return d_histo;
}

//--------HILLIS-STEELE SCAN----------
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
//Optimal step efficiency (histogram is a relatively small vector)
//Works on maximum 1024 (Pascal) elems vector.
// Single buffer work only with a warp.
// Double buffer only work with a block.
__global__ void scan_hillis_steele(unsigned int* d_out,const unsigned int* d_in, int size) {
	extern __shared__ unsigned int temp[];
	int tid = threadIdx.x;
	int pout = 0,pin=1;
	temp[tid] = tid>0? d_in[tid-1]:0; //exclusive scan
	__syncthreads();

	//double buffered
	for (int off = 1; off < size; off <<= 1) {
      // Using 1->0->1->0 for pin and pout because:
      // At every iteration the new output will be needed as an input.
      // SO we set the initial(not used anymore) input buffer as the new output buffer
      // and the new output buffer as the new input buffer.
		pout = 1 - pout; //Set 1 -> 0 -> 1 ...
		pin = 1 - pout; // Set 0 -> 1 -> 0 ...
		if (tid >= off) temp[size*pout + tid] = temp[size*pin + tid]+temp[size*pin + tid - off];
		else temp[size*pout + tid] = temp[size*pin + tid];
		__syncthreads();
	}
	d_out[tid] = temp[pout*size + tid];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
   // 1) find the minimum and maximum value in the input logLuminance channel
   // store in min_logLum and max_logLum
   int input_size = numRows * numCols;
   reduce(d_logLuminance, min_logLum, max_logLum, input_size);

   // 2) subtract them to find the range
   float logLumRange = max_logLum - min_logLum;

   // 3) generate a histogram of all the values in the logLuminance channel using
   //  the formula: bin ID = (lum[i] - lumMin) / lumRange * numBins
   unsigned int* d_histo = compute_histogram(d_logLuminance, numBins, input_size, min_logLum, logLumRange);

   // 4) Perform an exclusive scan (prefix sum) on the histogram to get
   //  the cumulative distribution of luminance values (this should go in the
   //  incoming d_cdf pointer which already has been allocated for you)       */
   scan_hillis_steele <<<1, numBins, 2*numBins*sizeof(unsigned int) >>> (d_cdf,d_histo, numBins);
}
