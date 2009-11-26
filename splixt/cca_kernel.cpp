#include "splixt.h"

#define __mins(val, x, y, dx, dy) x1 = (x) + (dx); y1 = (y) + (dy); pos = y1 * width + x1; if (x1 >= 0 && y1 >=0 && x1 < width && y1 < height && img[pos] == 0 && (val) > ll[pos]) (val) = ll[pos]

__global__ void cust_treshold (int *dst, int *img, int width, int height, int t)
{
   // Block index
   int x, y;
   x = blockIdx.x * blockDim.x + threadIdx.x;
   y = blockIdx.y * blockDim.y + threadIdx.y;

   dst[y * width + x] = img[y * width + x] > t ? 0 : 255;
}

__global__ void cust_cca_a_init (int *img, int width, int height, int *ll)
{
   int x = blockDim.x * blockIdx.x + threadIdx.x;
   int y = blockDim.y * blockIdx.y + threadIdx.y;
   int id = y * width + x;
   if (x >= width || y >= height || img[id] != 0)
      return;
   ll[id] = id;   
}

__global__ void cust_cca_a_update (int *img, int width, int height, int *ll, int *flag)
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
      ll[id] = minLabel;
      *flag = 1;
   }
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

__global__ void cust_cca_collect_labels (int *img, int width, int height, int *ll, int *ref, int *cnt)
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
      lpos = atomicAdd(cnt, 1);
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
   if (x >= ncc)
      return;
   c = cc + x;
   c->x0 = width;
   c->y0 = height;
   c->x1 = -1;
   c->y1 = -1;
   c->np = 0;
   c->tx = 0;
   c->ty = 0;
   c->g = -1;
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

__global__ void cust_cca_show_cc (int *img, int width, int height, ConComp *cc, int ncc, int* ll)
{
   int x = blockDim.x * blockIdx.x + threadIdx.x;
   int y = blockDim.y * blockIdx.y + threadIdx.y;   
   int xc, yc, i;
   int id = y * width + x;
   if (x >= width || y >= height || ll[id] < 0)
      return;   
   if (cc[ll[id]].g > 0)
      img[id] = 100; 
   else
      img[id] = 0xFF;
}

__global__ void cust_xy_project_y (ConComp* cc, int ncc, int* accu)
{
   int x = blockDim.x * blockIdx.x + threadIdx.x, i;
   ConComp *c;
   if (x >= ncc)
      return;
   c = cc + x;
   for (i = c->y0; i <= c->y1; ++i)
      atomicAdd(&accu[i], 1);
}

__global__ void cust_xy_label_accu_y (int* accu, int height, int* nl)
{                                    
   int treshold = 4, i, j, cnt, label = 0, gap = 1;

   cnt = 0;
   for (i = 0; i < height; ++i)
   {
      if (accu[i] < treshold)
      {
         accu[i] = -1;
         cnt++;
      }
      else
      {
         if (cnt > gap)
         {
            label++;
         }
         accu[i] = label;
         cnt = 0;
      }
   }
   *nl = label + 1;

}

__global__ void cust_xy_assign_y (ConComp* cc, int ncc, int* accu, int height)
{
   int x = blockDim.x * blockIdx.x + threadIdx.x, y, y0;
   ConComp *c;
   if (x >= ncc)
      return;
   c = cc + x;
   c->g = -1;
   y0 = c->ty / c->np;
   for (y = y0 - 3; y < y0 + 3; ++y)
      if (y >= 0 && y < height && c->g < accu[y])
         c->g = accu[y];
}

__global__ void cust_xy_project_x (ConComp* cc, int ncc, int* accu, int width)
{
   int x = blockDim.x * blockIdx.x + threadIdx.x, i;
   ConComp *c;
   if (x >= ncc)
      return;
   c = cc + x;
   if (c->g >= 0)
      for (i = c->x0; i <= c->x1; ++i)
      {
         atomicAdd(&accu[c->g * width + i], 70);      
         atomicMin(&accu[c->g * width + i], 255);
      }
}

__global__ void cust_xy_label_accu_x (int* accu, int width, int *xlabel)
{                                    
   int treshold = 1, i, j, cnt, label = -1, gap = 20;
   int x = blockDim.x * blockIdx.x + threadIdx.x;
   int *gaccu = accu + x * width;

   cnt = 0;
   for (i = 0; i < width; ++i)
   {
      if (gaccu[i] < treshold)
      {
         gaccu[i] = -1;       
         cnt++;
      }
      else
      {
         if (label < 0 || cnt > gap)
            label = atomicAdd(xlabel, 1);         
         else
            for (j = i - cnt; j < i; ++j)
               gaccu[j] = label;
         gaccu[i] = label;
         cnt = 0;
      }
   }
}

__global__ void cust_xy_assign_x (ConComp* cc, int ncc, int* accu, int width)
{
   int x = blockDim.x * blockIdx.x + threadIdx.x;
   ConComp *c;
   if (x >= ncc)
      return;
   c = cc + x;
   if (c->g >= 0)
      c->g = accu[c->g * width + c->tx / c->np];
}

__global__ void cust_cs_props_count(ConSet* css, int ncs, ConComp* cc, int ncc)
{
   int x = blockDim.x * blockIdx.x + threadIdx.x;
   ConComp *c;
   ConSet *cs;
   if (x >= ncc)
      return;
   c = cc + x;
   if (c->g >= ncs || c->g < 0)
      return;
   cs = css + c->g;
   atomicAdd(&cs->ncc, 1);
   atomicAdd(&cs->nop, c->np);
   atomicAdd(&cs->acc, (c->x1 - c->x0 + 1) * (c->y1 - c->y0 + 1));
   atomicMin(&cs->x0, c->x0);
   atomicMax(&cs->x1, c->x1);
   atomicMin(&cs->y0, c->y0);
   atomicMax(&cs->y1, c->y1);
}

__global__ void cust_cs_clear (ConSet* css, int ncs, int width, int height)
{
   int x = blockDim.x * blockIdx.x + threadIdx.x;
   ConSet *cs;
   if (x >= ncs)
      return;
   cs = css + x;
   cs->x0 = width;
   cs->y0 = height;
   cs->x1 = -1;
   cs->y1 = -1;
   cs->nop = 0;
   cs->ncc = 0;
   cs->acc = 0;
}

__global__ void cust_cca_show_cs (int *img, int width, int height, ConComp* cc, ConSet *css, int* ll, int ncs)
{
   int x = blockDim.x * blockIdx.x + threadIdx.x;
   int y = blockDim.y * blockIdx.y + threadIdx.y;   
   int label, g, i, incs = 0;
   ConSet *cs;
   int id = y * width + x;
   if (x >= width || y >= height || ll[id] < 0)
      return; 
   for (i = 0; i < ncs; ++i)
      if (x >= css[i].x0 && x <= css[i].x1 && y >= css[i].y0 && y <= css[i].y1)
         incs = 1;
   label = ll[id];
   g = cc[label].g;
   //if (incs)
   //   img[id] = 255; //170;
   //else
   //   img[id] = 255;
   if (g >= 0 && g < MAX_CC && css[g].is_text)
      if (incs)
         img[id] = 100;
      //else
      //   img[id] = 0;
}

__global__ void cust_cca_cs_is_text (ConSet *css, int ncs, int* img, int width, int height)
{
   int id = blockDim.x * blockIdx.x + threadIdx.x;
   int j, x, y, a, c;
   ConSet *cs;
   float r, d, r3, r5, r6;
   if (id >= ncs)      
      return;    
   cs = css + id;
   int w = cs->x1 - cs->x0 + 1;
   int h = cs->y1 - cs->y0 + 1;

   cs->is_text = 1;

   r = (float)w / h;
   if (r < 2.0f || 0.5f * r > cs->ncc || 8.0f * r < cs->ncc || cs->nop <= 3)
   {
      cs->is_text = 0;
      return;
   }

   r3 = (float)cs->acc / (w * h);
   if (r3 < 0.3f || r3 > 0.95f)
   {
      cs->is_text = 0;
      return;
   }

   a = 0;
   for (y = cs->y0; y <= cs->y1; ++y)
      for (x = cs->x0; x <= cs->x1; ++x)
         if (img[y * width + x] == 0)
            a++;      
   r5 = (float)a / (w * h);
   if (r5 < 0.1f || r5 > 0.8f)
   {
      cs->is_text = 0;
      return;
   }

   a = 0;
   c = 0;
   if (cs->y0 > 0)
   {
      c += w;
      for (x = cs->x0; x <= cs->x1; ++x)
         if (img[(cs->y0 - 1) * width + x] == 0)
            a++;      
   }
   if (cs->y1 < height - 1)
   {
      c += w;
      for (x = cs->x0; x <= cs->x1; ++x)
         if (img[(cs->y1 + 1) * width + x] == 0)
            a++;      
   }
   if (cs->x0 > 0)
   {
      c += h;
      for (y = cs->y0; y <= cs->y1; ++y)
         if (img[y * width + cs->x0 - 1] == 0)
            a++;      
   }
   if (cs->x1 < width - 1)
   {
      c += h;
      for (x = cs->x0; x <= cs->x1; ++x)
         if (img[y * width + cs->x0 + 1] == 0)
            a++;      
   }
   r6 = (float)a / c;
   if (r6 > 0.07f)
   {
      cs->is_text = 0;
      return;
   }
}
