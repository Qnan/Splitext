#include <stdio.h>
#include "cust.h"

#define BLOCK_SIZE 16
                                                                   
__global__ void cust_treshold (int *img, int width, int height, int t)
{
   // Block index
   int bx = blockIdx.x;
   int by = blockIdx.y;
   int x, y, x0, x1, y0, y1, dy;

   x0 = bx * BLOCK_SIZE;
   y0 = by * BLOCK_SIZE;
   x1 = (bx + 1) * BLOCK_SIZE;
   y1 = (by + 1) * BLOCK_SIZE;

   if (x1 >= width)
      x1 = width - 1;
   if (y1 >= height)
      y1 = height - 1;

   for (y = y0; y < y1; ++y)
      for (x = x0, dy = y * width; x < x1; ++x)
         img[dy + x] = img[dy + x] < t ? 0 : 255;
}

__global__ void cust_cca_d_init (int *img, int width, int height, int *ll, int *ref)
{
   int x = blockDim.x * blockIdx.x + threadIdx.x;
   int y = blockDim.y * blockIdx.y + threadIdx.y;
   int id = y * width + x;
   if (x >= width || y >= height || img[id] != 0)
      return;
   ll[id] = id;
   ref[id] = id;
}

#define __mins(val, x, y, dx, dy) x1 = (x) + (dx); y1 = (y) + (dy); pos = y1 * width + x1; if (x1 >= 0 && y1 >=0 && x1 < width && y1 < height && img[pos] == 0 && (val) > ll[pos]) (val) = ll[pos]

__global__ void cust_cca_d_scan (int *img, int width, int height, int *ll, int *ref, bool *flag)
{
   int x = blockDim.x * blockIdx.x + threadIdx.x;
   int y = blockDim.y * blockIdx.y + threadIdx.y;
   int id = y * width + x;
   if (x >= width || y >= height || img[id] != 0)
      return;
   int minLabel = ll[id];
   int x1, y1, pos;

   __mins(minLabel, x, y, -1, -1);
   __mins(minLabel, x, y,  0, -1);
   __mins(minLabel, x, y, +1, -1);
   __mins(minLabel, x, y, -1,  0);
   __mins(minLabel, x, y, +1,  0);
   __mins(minLabel, x, y, -1, +1);
   __mins(minLabel, x, y,  0, +1);
   __mins(minLabel, x, y, +1, +1);
   
   if (minLabel < ll[id])
   {
      atomicMin(&ref[ll[id]], minLabel);
      *flag = true;
   }
}

__global__ void cust_cca_d_resolve (int *img, int width, int height, int *ll, int *ref)
{
   int x = blockDim.x * blockIdx.x + threadIdx.x;
   int y = blockDim.y * blockIdx.y + threadIdx.y;
   int id = y * width + x;
   if (x >= width || y >= height || img[id] != 0)
      return;
   int r, label = ll[id];
   
   if (id == label)
   {
      do {
         r = label;
         label = ref[r];
      } while (r != label);
      ref[id] = label;
   }   
}

__global__ void cust_cca_d_relabel (int* img, int width, int height, int *ll, int *ref)
{
   int x = blockDim.x * blockIdx.x + threadIdx.x;
   int y = blockDim.y * blockIdx.y + threadIdx.y;
   int id = y * width + x;
   if (x >= width || y >= height || img[id] != 0)
      return;
   ll[id] = ref[ref[ll[id]]];
}

__global__ void cust_cca_d_display (int *img, int width, int height, int *ll)
{
   int x = blockDim.x * blockIdx.x + threadIdx.x;
   int y = blockDim.y * blockIdx.y + threadIdx.y;   
   int id = y * width + x;
   if (x >= width || y >= height || img[id] != 0)
      return;
   int label = ll[id];
   int color = 0, i;
   for (i = 0; i < 6; ++i)
      if (label & (1 << i))
         color |= 0xF << (4*i);
   img[id] = color;
}