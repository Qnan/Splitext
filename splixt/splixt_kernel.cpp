#include <stdio.h>
#include "splixt.h"

__global__ void splixt_treshold (int *img, int width, int height, int t)
{
   // Region index
   int rx = blockIdx.x;
   int ry = blockIdx.y;
   int x, y, x0, x1, y0, y1, dy;

   x0 = rx * BLOCK_SIZE;
   y0 = ry * BLOCK_SIZE;
   x1 = (rx + 1) * BLOCK_SIZE;
   y1 = (ry + 1) * BLOCK_SIZE;

   if (x1 >= width)
      x1 = width - 1;
   if (y1 >= height)
      y1 = height - 1;

   for (y = y0; y < y1; ++y)
      for (x = x0, dy = y * width; x < x1; ++x)
         img[dy + x] = img[dy + x] < t ? 0 : 255;
}

__global__ void splixt_region_split (Region *rr, int nx, int ny, int* img, int img_pitch)
{
   Region *r;
   Layer *layer;                
   int x, y, xi, yi, rx, ry, v, t0, t1, lid,
      v01, v02, v11, v12, v21, v22,
      t, tb, i;

   float bcv = 0, variance_treshold, d1, d2, max_variance,
      sigma, max_sigma;
   rx = blockIdx.x * blockDim.x + threadIdx.x;
   ry = blockIdx.y * blockDim.y + threadIdx.y;
   r = rr + ry * nx + rx;
   for (v = 0; v < MAX_HIST; ++v)
      r->hist[v] = 0;

   r->rx = rx;
   r->ry = ry;
   r->max_plane = 0;
   for (y = 0; y < REG_SIZE; ++y)
   {
      for (x = 0; x < REG_SIZE; ++x)
      {
         xi = rx * REG_SIZE + x;
         yi = ry * REG_SIZE + y;
         r->hist[img[yi * img_pitch + xi]]++;
      }
   }

   r->layer_count = 1;
   r->layers[0].t = 0;
   lid = 0;
   layer = r->layers + lid;
   t0 = layer->t;
   t1 = lid < r->layer_count - 1 ? r->layers[lid + 1].t : MAX_HIST;
   layer->v0 = layer->v1 = layer->v2 = 0;
   for (v = t0; v < t1; ++v)
   {
      layer->v0 += r->hist[v];
      layer->v1 += v * r->hist[v];
      layer->v2 += v * v * r->hist[v];
   }
   layer->avg = (float)layer->v1 / layer->v0;
   layer->stddev = (float)layer->v2 / layer->v0 - layer->avg * layer->avg;
   layer->q = 0;
   r->average = r->layers[0].avg;
   r->variance = r->layers[0].stddev;

   variance_treshold = separability_treshold * r->variance;
   lid = 0;
   do
   {  
      max_sigma = 0;
      // select layer to split
      for (i = 0; i < r->layer_count; ++i)
      {
         sigma = r->layers[i].stddev * r->layers[i].v0 / REG_SIZE_SQ;
         if (sigma > max_sigma)
         {
            max_sigma = sigma;
            lid = i;
         }
      }      

      // find optimal treshold
      layer = r->layers + lid;
      t0 = layer->t;
      t1 = lid < r->layer_count - 1 ? r->layers[lid + 1].t : MAX_HIST;

      v01 = 0;
      v02 = layer->v0;
      v11 = 0;
      v12 = layer->v1;
      v21 = 0;
      v22 = layer->v2;
      
      max_variance = 0;
      tb = -1;
      for (t = t0; t < t1; ++t)
      {
         if (v01 > 0 && v02 > 0)
         {
            d1 = (float)v11 / v01 - r->average;
            d2 = (float)v12 / v02 - r->average;
            bcv = (d1 * d1 * v01 + d2 * d2 * v02) / layer->v0;
            if (bcv > max_variance)
            {
               max_variance = bcv;
               tb = t;
            }     
         }
         v01 += r->hist[t];
         v02 -= r->hist[t];
         v11 += t * r->hist[t];
         v12 -= t * r->hist[t];
         v21 += t * t * r->hist[t];
         v22 -= t * t * r->hist[t];
      }

      if (tb < 0)
         break; // can't split layer

      if (r->layer_count == 1 && bcv < r->variance * homogeniety_separability_treshold && r->variance < homogeniety_variance_treshold)
         break; // homogenious region
                                    
      // split layer
      for (i = r->layer_count - 1; i > lid; --i)
         r->layers[i + 1] = r->layers[i];
      r->layer_count++;
      r->layers[lid + 1].t = tb;
      layer = r->layers + lid;
      t0 = layer->t;
      t1 = lid < r->layer_count - 1 ? r->layers[lid + 1].t : MAX_HIST;
      layer->v0 = layer->v1 = layer->v2 = 0;
      for (v = t0; v < t1; ++v)
      {
         layer->v0 += r->hist[v];
         layer->v1 += v * r->hist[v];
         layer->v2 += v * v * r->hist[v];
      }
      layer->avg = (float)layer->v1 / layer->v0;
      layer->stddev = (float)layer->v2 / layer->v0 - layer->avg * layer->avg;
      layer->q = 0;
      lid++;
      layer = r->layers + lid;
      t0 = layer->t;
      t1 = lid < r->layer_count - 1 ? r->layers[lid + 1].t : MAX_HIST;
      layer->v0 = layer->v1 = layer->v2 = 0;
      for (v = t0; v < t1; ++v)
      {
         layer->v0 += r->hist[v];
         layer->v1 += v * r->hist[v];
         layer->v2 += v * v * r->hist[v];
      }
      layer->avg = (float)layer->v1 / layer->v0;
      layer->stddev = (float)layer->v2 / layer->v0 - layer->avg * layer->avg;
      layer->q = 0;
   } while (bcv < variance_treshold && r->layer_count < MAX_LAYERS);
                     
   // write split results
   for (lid = 0; lid < r->layer_count; ++lid)
   {
      t0 = r->layers[lid].t;
      t1 = lid < r->layer_count - 1 ? r->layers[lid + 1].t : MAX_HIST;
      for (y = 0; y < REG_SIZE; ++y)
      {
         for (x = 0; x < REG_SIZE; ++x)
         {
            xi = rx * REG_SIZE + x;
            yi = ry * REG_SIZE + y;
            v = img[yi * img_pitch + xi];
            if (v >= t0 && v < t1)      
               img[yi * img_pitch + xi] = (int)r->layers[lid].avg;
         }
      }
   }
}