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

__global__ void splixt2_calc_hist (int* histogram, int *img, int width, int height)
{  
   int x, y;

   x = blockIdx.x * blockDim.x + threadIdx.x;
   y = blockIdx.y * blockDim.y + threadIdx.y;
   if (x >= width || y >= height)
      return;

   atomicAdd(&histogram[img[y * width + x]], 1);
}

__global__ void splixt_region_split (Region* rr, int nx, int ny, int *img, int width)
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
         r->hist[img[yi * width + xi]]++;
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
   layer->q = -1;
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
      layer->q = -1;
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
      layer->q = -1;
   } while (bcv < variance_treshold && r->layer_count < MAX_LAYERS);
                     
   // write split results
   //for (lid = 0; lid < r->layer_count; ++lid) {
   //   t0 = r->layers[lid].t;
   //   t1 = lid < r->layer_count - 1 ? r->layers[lid + 1].t : MAX_HIST;
   //   for (y = 0; y < REG_SIZE; ++y) {
   //      for (x = 0; x < REG_SIZE; ++x) {
   //         xi = rx * REG_SIZE + x;
   //         yi = ry * REG_SIZE + y;
   //         v = img[yi * width + xi];
   //         if (v >= t0 && v < t1)      
   //            g->out_img[yi * width + xi] = (int)r->layers[lid].avg;
   //      }
   //   }
   //}
}

__global__ void splixt_calc_mountine_initial (Region* rr, int nx, int ny, Mountine* m)
{
   Region *r, *r2;
   Layer *layer;
   int rx, ry, x, y, i, j, j0;
   float max, mountine;

   rx = blockIdx.x * blockDim.x + threadIdx.x;
   ry = blockIdx.y * blockDim.y + threadIdx.y;
   if (rx >= nx || ry >= ny)
      return;
   r = rr + ry * nx + rx;

   // calculate initial mountine function
   max = 0;
   for (j = 0; j < r->layer_count; ++j)
   {
      mountine = 0;
      layer = r->layers + j;
      for (y = 0; y < ny; ++y) {
         for (x = 0; x < nx; ++x) {
            r2 = rr + y * nx + x;
            for (i = 0; i < r2->layer_count; ++i) {
               mountine += expf(-3 * fabs(r2->layers[i].avg - layer->avg));
            }
         }
      }
      layer->mountine = mountine;
      if (mountine > max)
      {
         max = mountine;
         j0 = j;
      }
   }

   while (m->mountine < max)
   {
      // acquire lock
      if (!atomicCAS(&m->lock, 0, 1))
      {
         // check again
         if (m->mountine < max)
         {
            m->rx = rx;
            m->ry = ry;
            m->lid = j0;
            m->mountine = max;
            m->avg = r->layers[j0].avg;
         }
         // release lock
         m->lock = 0;
      }
   }
}

__global__ void splixt_calc_mountine_update (Region* rr, int nx, int ny, Mountine* m, Mountine* mold)
{
   Region *r;
   Layer *layer;
   int rx, ry, j, j0;
   float max;

   rx = blockIdx.x * blockDim.x + threadIdx.x;
   ry = blockIdx.y * blockDim.y + threadIdx.y;
   if (rx >= nx || ry >= ny)
      return;
   r = rr + ry * nx + rx;

   // update mountine function value
   max = 0;
   for (j = 0; j < r->layer_count; ++j)
   {
      layer = r->layers + j;
      layer->mountine -= mold->mountine * expf(-0.4f * fabs(mold->avg - layer->avg));
      if (layer->mountine > max)
      {
         max = layer->mountine;
         j0 = j;
      }
   }

   while (m->mountine < max)
   {
      // acquire lock
      if (!atomicCAS(&m->lock, 0, 1))
      {
         // check again
         if (m->mountine < max)
         {
            m->rx = rx;
            m->ry = ry;
            m->lid = j0;
            m->mountine = max;
            m->avg = r->layers[j0].avg;
         }
         // release lock
         m->lock = 0;
      }
   }
}

__global__ void splixt_planes_init (Region* rr, int nx, int ny, Plane* pp, Mountine* mm)
{
   Mountine *m;
   Region *r;
   Plane *p;
   Layer *l;
   int q;
   
   q = blockIdx.x;             
   m = mm + q;
   p = pp + q;
   r = rr + m->ry * nx + m->rx;
   l = r->layers + m->lid;

   l->q = q;
   r->pln_dst[q] = 0;
   p->v0 = l->v0;
   p->v1 = l->v1;
   p->v2 = l->v2;
   p->lock = 0;
   p->v0 += 1;
}  

__global__ void splixt_plane_construct (Region* rr, int nx, int ny, Plane* pp, int *flag, int* img, int width)
{
   Region *r, *r2, *ra, *rb;
   Layer *l, *l2;
   Plane *p;
   int rx, ry, j, i, k, q0, dx, dy, 
      xa, xb, ya, yb, s, la, lb, va, vb, n2,
      ta0, ta1, tb0, tb1;
   float min = 0, diff, diff2, sigma, single;
   int nei[4], neicnt = 0;

   rx = blockIdx.x * blockDim.x + threadIdx.x;
   ry = blockIdx.y * blockDim.y + threadIdx.y;
   if (rx >= nx || ry >= ny)
      return;
   r = rr + ry * nx + rx;

   if (rx > 0)
      nei[neicnt++] = ry * nx + rx - 1;
   if (rx < nx - 1)
      nei[neicnt++] = ry * nx + rx + 1;
   if (ry > 0)
      nei[neicnt++] = (ry - 1) * nx + rx;
   if (ry < ny - 1)
      nei[neicnt++] = (ry + 1) * nx + rx; 

   // try to assign every layer to some plane
   for (j = 0; j < r->layer_count; ++j)
   {
      // find the most similiar layer among the neighbors
      l = r->layers + j;
      if (l->q >= 0)
         continue; // skip already assigned
      q0 = -1;
      for (i = 0; i < neicnt; ++i)
      {
         r2 = rr + nei[i];
         for (k = 0; k < r2->layer_count; ++k)
         {
            l2 = r2->layers + k;
            if (l2->q < 0)
               continue; // skip unassigned neighbors

            if (r2->rx == r->rx)
            {
               dx = 1;
               dy = 0;
               xa = xb = r->rx * REG_SIZE;
               if (r2->ry > r->ry)
               {
                  ra = r;
                  rb = r2;
                  la = j;
                  lb = k;
                  ya = r2->ry * REG_SIZE - 1;
                  yb = r2->ry * REG_SIZE;
               }
               else
               {
                  ra = r2;
                  rb = r;
                  la = k;
                  lb = j;
                  ya = r->ry * REG_SIZE;
                  yb = r->ry * REG_SIZE - 1;
               }
            }
            else
            {
               dx = 0;
               dy = 1;
               ya = yb = r->ry * REG_SIZE;
               if (r2->rx > r->rx)
               {
                  ra = r;
                  rb = r2;
                  la = j;
                  lb = k;
                  xa = r2->rx * REG_SIZE - 1;
                  xb = r2->rx * REG_SIZE;
               }
               else
               {
                  ra = r2;
                  rb = r;
                  la = k;
                  lb = j;
                  xa = r->rx * REG_SIZE;
                  xb = r->rx * REG_SIZE - 1;
               }
            }
            ta0 = ra->layers[la].t;
            ta1 = la < ra->layer_count - 1 ? ra->layers[la + 1].t : MAX_HIST;
            tb0 = rb->layers[lb].t;
            tb1 = lb < rb->layer_count - 1 ? rb->layers[lb + 1].t : MAX_HIST;
            diff2 = 0;
            n2 = 0;
            for (s = 0; s < REG_SIZE; ++s, xa += dx, xb += dx, ya += dy, yb += dy)
            {  
               va = img[ya * width + xa];
               vb = img[yb * width + xb];
               if (va >= ta0 && va < ta1 && vb >= tb0 && vb < tb1)
               {
                  diff2 += abs(va - vb);
                  n2++;
               }
            }          
            if (n2 > 0)
               diff2 /= n2;
            else
               diff2 = 255;            

            diff = fabs(l->avg - l2->avg);
            if (diff < diff2)
               diff = diff2;
            sigma = sqrtf(l2->stddev);
            if (q0 < 0 || diff < min)
            {
               q0 = l2->q;
               min = diff;
            }
         }
      }
      
      // check similiarity
      if (q0 >= 0)
      {
         sigma += sqrtf(l->stddev);
         if (sigma < 1) sigma = 1;
         single = min / sigma;

         if (single < 1.5)
         {  
            l->q = q0;
            r->pln_dst[q0] = 0;
            p = pp + q0;
         
            // lock
            while (!atomicCAS(&p->lock, 0, 1))
               ;
            p->v0 += l->v0;
            p->v1 += l->v1;
            p->v2 += l->v2;
            *flag = 1;
            // unlock
            p->lock = 0;
         }
      }
   }            
}


__global__ void splixt_seed_show (int* out, int* img, int width, Region* rr, int nx, int ny, Mountine* m, int pc)
{
   Region *r;
   int rx, ry, x, y, i, t0, t1, v;

   rx = blockIdx.x * blockDim.x + threadIdx.x;
   ry = blockIdx.y * blockDim.y + threadIdx.y;
   if (rx >= nx || ry >= ny)
      return;
   r = rr + ry * nx + rx;

   for (i = 0; i < pc; ++i) {
      if (m[i].rx == rx && m[i].ry == ry) {
         t0 = r->layers[i].t;
         t1 = i < r->layer_count - 1 ? r->layers[i + 1].t : MAX_HIST;

         for (x = 0; x < REG_SIZE; ++x) {
            for (y = 0; y < REG_SIZE; ++y) {
               v = img[(ry * REG_SIZE + y) * width + rx * REG_SIZE + x];
               if (v >= t0 && v < t1)
                  out[(ry * REG_SIZE + y) * width + rx * REG_SIZE + x] = r->layers[m[i].lid].avg;
            }
         }
      }
   }
}

__global__ void splixt_binarize (int* bp, int* img, int width, Region* rr, int nx, int ny, Plane* pp, int pid)
{
   Region *r;
   Layer *l;
   int rx, ry, x, y, i, t0, t1, v;

   rx = blockIdx.x * blockDim.x + threadIdx.x;
   ry = blockIdx.y * blockDim.y + threadIdx.y;
   if (rx >= nx || ry >= ny)
      return;
   r = rr + ry * nx + rx;

   for (x = 0; x < REG_SIZE; ++x)
      for (y = 0; y < REG_SIZE; ++y)
         bp[(ry * REG_SIZE + y) * width + rx * REG_SIZE + x] = 0xFF;
   for (i = 0; i < r->layer_count; ++i)
   {
      l = r->layers + i;
      if (l->q == pid)
      {        
         t0 = l->t;
         t1 = i < r->layer_count - 1 ? r->layers[i + 1].t : MAX_HIST;
         for (x = 0; x < REG_SIZE; ++x)
         {
            for (y = 0; y < REG_SIZE; ++y)
            {                                                            
               v = img[(ry * REG_SIZE + y) * width + rx * REG_SIZE + x];
               if (v >= t0 && v < t1)
                  bp[(ry * REG_SIZE + y) * width + rx * REG_SIZE + x] = 0;
            }                    
         }
      }
   }
}

__global__ void splixt_plane_show (int* out, int* img, int width, Region* rr, int nx, int ny, Plane* pp, int pc)
{
   Region *r;
   Layer *l;
   int rx, ry, x, y, i, t0, t1, v, c;

   rx = blockIdx.x * blockDim.x + threadIdx.x;
   ry = blockIdx.y * blockDim.y + threadIdx.y;
   if (rx >= nx || ry >= ny)
      return;
   r = rr + ry * nx + rx;

   for (i = 0; i < r->layer_count; ++i)
   {
      l = r->layers + i;
      if (l->q >= 0)
      {        
         c = l->q * 255 / pc;
         t0 = l->t;
         t1 = i < r->layer_count - 1 ? r->layers[i + 1].t : MAX_HIST;
         for (x = 0; x < REG_SIZE; ++x)
         {
            for (y = 0; y < REG_SIZE; ++y)
            {                                                            
               v = img[(ry * REG_SIZE + y) * width + rx * REG_SIZE + x];
               if (v >= t0 && v < t1)
                  out[(ry * REG_SIZE + y) * width + rx * REG_SIZE + x] = c;
            }                    
         }
      }
   }
}

__global__ void splixt_img_clear (int* img, int width, int nx, int ny)
{
   int rx, ry, x, y;

   rx = blockIdx.x * blockDim.x + threadIdx.x;
   ry = blockIdx.y * blockDim.y + threadIdx.y;
   if (rx >= nx || ry >= ny)
      return;

   for (x = 0; x < REG_SIZE; ++x)
      for (y = 0; y < REG_SIZE; ++y)
         img[(ry * REG_SIZE + y) * width + rx * REG_SIZE + x] = 0xFF;
}




