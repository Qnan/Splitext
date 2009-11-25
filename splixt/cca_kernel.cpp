#include "splixt.h"

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

__global__ void cust_cca_d_scan (int *img, int width, int height, int *ll, int *ref, int *flag)
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
      *flag = 1;
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
   int color = 0;
   color = label;
   //for (i = 0; i < 6; ++i)
   //   if (label & (1 << i))
   //      color |= 0xF << (4*i);
   img[id] = color;
}

__global__ void cust_cca_collect_labels (int *img, int width, int height, int *ll, int *ref, unsigned int *cnt)
{
   int x = blockDim.x * blockIdx.x + threadIdx.x;
   int y = blockDim.y * blockIdx.y + threadIdx.y;   
   int id = y * width + x;
   int label, lpos;
   if (x >= width || y >= height || img[id] != 0)
      return;
   label = ll[id];   
   if (label == id)
   {
      lpos = atomicInc(cnt, width * height);
      //ll[lpos] = label;
      ref[label] = lpos;
   }
}

__global__ void cust_cca_d_relabel_1 (int* img, int width, int height, int *ll, int *ref)
{
   int x = blockDim.x * blockIdx.x + threadIdx.x;
   int y = blockDim.y * blockIdx.y + threadIdx.y;
   int id = y * width + x;
   if (x >= width || y >= height || img[id] != 0)
      return;
   ll[id] = ref[ll[id]];
}

__global__ void cust_cca_label_clear (ConComp* cc, int ncc, int width, int height)
{
   int x = blockDim.x * blockIdx.x + threadIdx.x;
   ConComp *c;
   if (x > ncc)
      return;
   c = cc + x;
   c->x0 = width;
   c->y0 = height;
   c->x1 = -1;
   c->y1 = -1;
   c->np = 0;
   c->tx = 0;
   c->ty = 0;
}

__global__ void cust_cca_labels_calc_props (int *img, int width, int height, int *ll, ConComp* cc)
{
   ConComp *c;
   int x = blockDim.x * blockIdx.x + threadIdx.x;
   int y = blockDim.y * blockIdx.y + threadIdx.y;   
   int id = y * width + x;
   if (x >= width || y >= height || img[id] != 0)
      return;
   c = cc + ll[id];
   if (c->x0 > x)
      atomicMin(&c->x0, x);
   if (c->x1 < x)
      atomicMax(&c->x1, x);
   if (c->y0 > y)
      atomicMin(&c->y0, y);
   if (c->y1 < y)
      atomicMax(&c->y1, y);
   atomicAdd(&c->np, 1);
   atomicAdd(&c->tx, x);
   atomicAdd(&c->ty, y);
}
