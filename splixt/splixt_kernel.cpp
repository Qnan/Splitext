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
   if (rx >= nx || ry >= ny)
      return;

   r = rr + ry * nx + rx;
   for (v = 0; v < MAX_HIST; ++v)
      r->hist[v] = 0;

   r->rx = rx;
   r->ry = ry;
   r->max_plane = 0;
   for (y = 0; y < REG_SIZE; ++y) {
      for (x = 0; x < REG_SIZE; ++x) {
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
   for (v = t0; v < t1; ++v) {
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
   do {  
      max_sigma = 0;
      // select layer to split
      for (i = 0; i < r->layer_count; ++i) {
         sigma = r->layers[i].stddev * r->layers[i].v0 / REG_SIZE_SQ;
         if (sigma > max_sigma) {
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
      for (t = t0; t < t1; ++t) {
         if (v01 > 0 && v02 > 0) {
            d1 = (float)v11 / v01 - r->average;
            d2 = (float)v12 / v02 - r->average;
            bcv = (d1 * d1 * v01 + d2 * d2 * v02) / layer->v0;
            if (bcv > max_variance) {
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
      for (v = t0; v < t1; ++v) {
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
   //for (lid = 0; lid < r->layer_count; ++lid) {
   //   t0 = r->layers[lid].t;
   //   t1 = lid < r->layer_count - 1 ? r->layers[lid + 1].t : MAX_HIST;
   //   for (y = 0; y < REG_SIZE; ++y) {
   //      for (x = 0; x < REG_SIZE; ++x) {
   //         xi = rx * REG_SIZE + x;
   //         yi = ry * REG_SIZE + y;
   //         v = img[yi * img_pitch + xi];
   //         if (v >= t0 && v < t1)      
   //            img[yi * img_pitch + xi] = (int)r->layers[lid].avg;
   //      }
   //   }
   //}
}

__global__ void splixt_plane_construct (Global *g, Region *rr, int nx, int ny, int* img, int img_pitch, Mountine* mnts)
{
   Region *r, *r2;
   Layer *layer;                
   Mountine *mnt;
   int x, y, xi, yi, rx, ry, lid, i, i0, nn, q, mnt2, d, j, k, m, t0, t1, v;
   int nei[4], neicnt = 0;
   float mountine, max_mountine, avg, first_mountine;
   rx = blockIdx.x * blockDim.x + threadIdx.x;
   ry = blockIdx.y * blockDim.y + threadIdx.y;
   if (rx >= nx || ry >= ny)
      return;
   r = rr + ry * nx + rx;
   nn = nx * ny;

   if (rx > 0)
      nei[neicnt++] = ry * nx + rx - 1;
   if (rx < nx - 1)
      nei[neicnt++] = ry * nx + rx + 1;
   if (ry > 0)
      nei[neicnt++] = (ry - 1) * nx + rx;
   if (ry < ny - 1)
      nei[neicnt++] = (ry + 1) * nx + rx;

   // calculate initial mountine function
   max_mountine = 0;
   for (lid = 0; lid < r->layer_count; ++lid)
   {
      mountine = 0;
      layer = r->layers + lid;
      for (y = 0; y < ny; ++y) {
         for (x = 0; x < nx; ++x) {
            r2 = rr + y * nx + x;
            for (i = 0; i < r2->layer_count; ++i) {
               mountine += expf(-fabs(r2->layers[i].avg - layer->avg));
            }
         }
      }
      layer->mountine = mountine;
      if (mountine > max_mountine)
         max_mountine = mountine;
   }

   // init plane distances
   for (i = 0; i < MAX_PLANES; ++i)
      r->pln_dst[i] = -1;

   __syncthreads();

   // select seeds
   q = 1;
   while (1)
   {
      if (q >= MAX_PLANES)
         break;
      mnt = mnts + ry * nx + rx;
      mnt->mountine = (int)(max_mountine / nn * INT_MAX / nn) * nn + ry * nx + rx;
      
      // find maximum mountine
      do {  
         __syncthreads();
         if (rx == 0 && ry == 0)
            g->change_flag = 0;
         __syncthreads();
         for (i = 0; i < neicnt; ++i)
         {
            mnt2 = mnts[nei[i]].mountine;
            if (mnt2 > mnt->mountine)
            {
               mnt->mountine = mnt2;
               g->change_flag = 1;
            }
         }
         __syncthreads();
      } while (g->change_flag);

      // max mountine region location
      r2 = rr + (mnt->mountine % nn);
      
      // find max mountine layer
      i0 = 0;
      for (i = 0; i < r2->layer_count; ++i)
         if (r2->layers[i].mountine > r2->layers[i0].mountine)
            i0 = i;
      avg = r2->layers[i0].avg;
      mountine = r2->layers[i0].mountine;
      
      if (rx == 0 && ry == 0)
      {
         r2->layers[i0].q = q;
         r2->pln_dst[q] = 0;
      }

      if (q == 1)
         first_mountine = mountine;
      q++;

      if (mountine < mountine_ratio_treshold * first_mountine)
         break;

      __syncthreads();

      max_mountine = 0;
      for (i = 0; i < r->layer_count; ++i)
      {
         r->layers[i].mountine -= mountine * expf(-fabs(avg - r->layers[i].avg));
         if (r->layers[i].mountine > max_mountine)
            max_mountine = r->layers[i].mountine;
      }

      __syncthreads();
      
      if (rx == 0 && ry == 0)           
         r2->layers[i0].mountine = 0;

      __syncthreads();
   }

   if (rx == 0 && ry == 0)
      for (i = 0; i < MAX_PLANES; ++i)
         g->pln_lock[i] = 0;

   __syncthreads();

   // extend planes
   do
   {
      __syncthreads();
      if (rx == 0 && ry == 0)
         g->change_flag_2 = 0;

      do {
         __syncthreads();
         if (rx == 0 && ry == 0)
            g->change_flag = 0;
         __syncthreads();

         for (i = 0; i < neicnt; ++i)
         {
            for (j = 0; j < q; ++j)
            {
               d = rr[nei[i]].pln_dst[j];
               if (d >= 0 && (r->pln_dst[j] < 0 || d + 1 < r->pln_dst[j]))
               {
                  r->pln_dst[j] = d + 1;
                  g->change_flag = 1;
               }
            }
         }
         __syncthreads();
      } while (g->change_flag);

      for (i = 0; i < neicnt; ++i)
      {
         r2 = rr + nei[i];
         for (j = 0; j < r2->layer_count; ++j)
         {
            d = r2->layers[j].q;
            if (r->pln_dst[d] > 0) {
               for (k = 0; k < r->layer_count; ++k) {
                  if (fabs(r->layers[k].avg - r2->layers[j].avg) < 5.0f) {
                     r->layers[k].q = d;
                     r->pln_dst[d] = 0;
                     g->change_flag_2 = 1;
                     break;
                  }
               }
            }
         }
      }

      __syncthreads();
   } while (g->change_flag_2);
   //i = 0;
   //while (!atomicCAS(g->pln_lock + i, 0, 1))
   //   ;
   //atomicCAS(g->pln_lock + i, 1, 0);

   // write split results
   for (lid = 0; lid < r->layer_count; ++lid) {
      t0 = r->layers[lid].t;
      t1 = lid < r->layer_count - 1 ? r->layers[lid + 1].t : MAX_HIST;
      for (y = 0; y < REG_SIZE; ++y) {
         for (x = 0; x < REG_SIZE; ++x) {
            xi = rx * REG_SIZE + x;
            yi = ry * REG_SIZE + y;
            v = img[yi * img_pitch + xi];
            if (v >= t0 && v < t1 && r->layers[lid].q == 1)      
               img[yi * img_pitch + xi] = 0;
            //else
            //   img[yi * img_pitch + xi] = 0xFF;
         }
      }
   }

}