#include "image.h"
#include "segment.h"
#include "malloc.h"
#include "memory.h"
#include "math.h"

static const int block_size = 120;
static const float separability_treshold = 0.9f; // 0.92f
static const float homogeniety_separability_treshold = 0.6f;
static const float homogeniety_variance_treshold = 11.0f;
static const float mountine_exp = 5.4f;
static const float mountine_ratio_treshold = 0.45f;

float fsqr(float val)
{
   return val * val;
}

#define MAX_LAYERS 16
#define MAX_HIST 256

typedef struct tagLayer{
   int t; // lower treshold border
   unsigned long v0; // total intensity
   unsigned long v1; // sum of v * h[v]
   unsigned long v2; // sum of v * v * h[v]
   float avg;
   float stddev;

   int q;
   float mountine;
}Layer;

typedef struct tagBlock{
   Layer layers[MAX_LAYERS];
   int histogram[MAX_HIST];
   int layer_count;
   int bx;
   int by; 
   int max_plane;
   float average;
   float variance;
}Block;

int layer_calc_props (Block* block, int layer_id)
{
   int v, t0, t1;
   Layer* layer = block->layers + layer_id;
   
   t0 = layer->t;
   t1 = layer_id < block->layer_count - 1 ? block->layers[layer_id + 1].t : MAX_HIST;

   layer->v0 = layer->v1 = layer->v2 = 0;
   for (v = t0; v < t1; ++v)
   {                                                 
      layer->v0 += block->histogram[v];
      layer->v1 += v * block->histogram[v];
      layer->v2 += v * v * block->histogram[v];
   }  
   layer->avg = (float)layer->v1 / layer->v0;
   layer->stddev = (float)layer->v2 / layer->v0 - layer->avg * layer->avg;
   layer->q = 0;
   return 0;
}

void block_prepare (Block* block, Image* img, int bx, int by)
{
   int x, y, xi, yi;
   memset(block->histogram, 0, MAX_HIST * sizeof(int));

   block->bx = bx;
   block->by = by;
   block->max_plane = 0;
   for (y = 0; y < block_size; ++y)
      for (x = 0; x < block_size; ++x)
      {
         xi = bx * block_size + x;
         yi = by * block_size + y;
         if (xi < img->width && yi < img->height)
            block->histogram[img->data[yi * img->width + xi]]++;
      }

   block->layer_count = 1;
   block->layers[0].t = 0;
   layer_calc_props(block, 0);
   block->average = block->layers[0].avg;
   block->variance = block->layers[0].stddev;
}

float block_calc_between_class_variance (Block* block)
{
   int i;
   Layer *layer;
   float bcv = 0, d;
   for (i = 0; i < block->layer_count; ++i)
   {
      layer = block->layers + i;
      d = layer->avg - block->average;
      bcv += d * d * layer->v0 / (block_size * block_size);
   }
   return bcv;
}

int block_find_treshold (Block* block, int layer_id, float *pbcv)
{
   int t, t_best = -1, t0, t1;
   float max_variance = 0, bcv, d1, d2;
   int total_intensity_0, total_intensity_1, object_pixels_0, object_pixels_1;
   Layer* layer = block->layers + layer_id;

   t0 = layer->t;
   t1 = layer_id < block->layer_count - 1 ? block->layers[layer_id + 1].t : MAX_HIST;

   total_intensity_0 = 0;
   total_intensity_1 = layer->v1;
   object_pixels_0 = 0;
   object_pixels_1 = layer->v0;

   for (t = t0; t < t1; ++t)
   {
      if (object_pixels_0 > 0 && object_pixels_1 > 0)
      {
         d1 = (float)total_intensity_0 / object_pixels_0 - block->average;
         d2 = (float)total_intensity_1 / object_pixels_1 - block->average;
         bcv = (d1 * d1 * object_pixels_0 + d2 * d2 * object_pixels_1) / layer->v0;
         if (bcv > max_variance)
         {
            max_variance = bcv;
            t_best = t;
         }
      }
      total_intensity_0 += t * block->histogram[t];
      total_intensity_1 -= t * block->histogram[t];
      object_pixels_0 += block->histogram[t];
      object_pixels_1 -= block->histogram[t];
   }
   if (pbcv != 0)
      *pbcv = max_variance;

   return t_best;
}

int layer_split (Block* block, int layer_id, int t, float* pbcv)
{     
   int i;
   float bcv = 0;

   if (block->layer_count >= MAX_LAYERS)
      return 1;

   if (t < 0)       
      t = block_find_treshold(block, layer_id, &bcv);
   if (t < 0)
      return 1;

   for (i = block->layer_count - 1; i > layer_id; --i)
      block->layers[i + 1] = block->layers[i];

   block->layer_count++;
   block->layers[layer_id + 1].t = t;

   layer_calc_props(block, layer_id);
   layer_calc_props(block, layer_id + 1);
   if (pbcv != NULL)
      *pbcv = bcv;
   return 0;
}

int block_get_max_sigma_layer (Block* block)
{
   int i, i0 = -1;
   float max_sigma = 0, sigma;
   for (i = 0; i < block->layer_count; ++i)
   {
      sigma = block->layers[i].stddev * block->layers[i].v0 / block_size / block_size;
      if (sigma > max_sigma)
      {
         max_sigma = sigma;
         i0 = i;
      }
   }   
   return i0;
}

int block_split (Block* block)
{
   int v, t;
   float variance_treshold, bcv;//, bcv1;

   variance_treshold = block->variance * separability_treshold;
   t = block_find_treshold(block, 0, &bcv);

   if (bcv < block->variance * homogeniety_separability_treshold && block->variance < homogeniety_variance_treshold)
      return 0;
   layer_split(block, 0, t, 0);      
   while (1)
   {
      v = block_get_max_sigma_layer(block);
      if (layer_split(block, v, -1, &bcv))
         break;
      //bcv1 = block_calc_between_class_variance(block);
      if (bcv > variance_treshold || block->layer_count >= MAX_LAYERS)
         break;
   }

   return 0;
}

int block_to_image (Image* img, Block* block, int layer_id, Image* source_image)
{
   int x,y,t0,t1,v;
   image_init(img, block_size, block_size);

   t0 = block->layers[layer_id].t;
   t1 = layer_id < block->layer_count - 1 ? block->layers[layer_id + 1].t : MAX_HIST;

   for (y = 0; y < block_size; ++y)
      for (x = 0; x < block_size; ++x)
      {           
         v = source_image->data[(block->by * block_size + y) * source_image->width + block->bx *  block_size + x];
         if (v >= t0 && v < t1)
            img->data[y * block_size + x] = v;
         else
            img->data[y * block_size + x] = 0xFF;
      }

   return 0;
}

void plane_to_image (Image* img, Block* blocks, Image* simg, int q, int nx, int ny)
{
   int x, y, x1, y1, bx, by, t0, t1, v, i, pos;
   Block *block;

   image_init(img, simg->width, simg->height);

   for (by = 0; by < ny; ++by)
   {
      y1 = by < ny - 1 ? block_size : simg->height - by * block_size;
      for (bx = 0; bx < nx; ++bx)
      {
         block = blocks + by * nx + bx;
         for (i = 0; i < block->layer_count && block->layers[i].q != q; ++i)
            ;
         if (i == block->layer_count)
            continue;           

         t0 = block->layers[i].t;
         t1 = i < block->layer_count - 1 ? block->layers[i + 1].t : MAX_HIST;

         x1 = bx < nx - 1 ? block_size : simg->width - bx * block_size;
         for (y = 0; y < y1; ++y)
            for (x = 0; x < x1; ++x)
            {
               pos = (by * block_size + y) * simg->width + bx * block_size + x;
               v = simg->data[pos];
               if (v >= t0 && v < t1)
                  img->data[pos] = v / 2;
            }
      }
   }
}

int layer_encode (int bx, int by, int nx, int k)
{                       
   return (by * nx + bx) * MAX_LAYERS + k;
}

Layer* layer_get (Block* blocks, int code, int nx)
{                       
   int k, bx, by;

   k = code % MAX_LAYERS;
   code /= MAX_LAYERS;
   bx = code % nx;
   by = code / nx;

   return &blocks[by * nx + bx].layers[k];
}

void layer_calc_mountine (Layer** pool, int layer_count, Layer* layer)
{
   int i;
   float mountine;
                                                       
   mountine = 0;
   for (i = 0; i < layer_count; ++i)
      if (pool[i] != layer && !pool[i]->q)
         mountine += expf(-/*mountine_exp*/ fabs(pool[i]->avg - layer->avg));
   layer->mountine = mountine;
}                                                       

void block_assign_plane(Image* image, Block* blocks, int bx, int by, int nx, int ny, int q, int* changed)
{
   int i, j, x, y, c1, c2, sx, sy, nsc, diff, x1, x2, y1, y2, j0;
   float side_measure, min_side_measure;
   Block *block = blocks + by * nx + bx, *block2;
   Layer *layer1, *layer2;

   if (block->max_plane >= q)
      return;

   min_side_measure = -1;
   block2 = blocks + by * nx + bx - 1;
   if (bx > 0 && block2->max_plane == q)
   {
      sx = bx * block_size;
      sy = by * block_size;
      x1 = sx;
      x2 = sx - 1;
      y2 = block_size;
      if (y2 > image->height - sy)
         y2 = image->height - sy;
      nsc = 0;
      diff = 0;
      for (i = 0; i < block2->layer_count && block2->layers[i].q != q; ++i)
         ;

      layer2 = block2->layers + i;
      for (j = 0; j < block->layer_count; ++j)
      {
         layer1 = block->layers + j;
         if (layer1->q)
            continue;
         for (y = 0; y < y2; ++y)
         {
            c1 = image->data[(sy + y) * image->width + x1];
            c2 = image->data[(sy + y) * image->width + x2];
            if (c1 >= layer1->t && (j == block->layer_count - 1 || c1 < block->layers[j + 1].t) &&
               c2 >= layer2->t && (i == block2->layer_count - 1 || c2 < block2->layers[i + 1].t))
            {
               diff += abs(c1 - c2);
               nsc++;
            }                  
         }
         if (nsc > 0)
         {
            side_measure = (float)diff / nsc;
            if (min_side_measure < 0 || side_measure < min_side_measure)
            {
               j0 = j;
               min_side_measure = side_measure;
            }
         }
      }
   }

   block2 = blocks + by * nx + bx + 1;
   if (bx < nx - 1 && block2->max_plane == q)
   {
      sx = (bx + 1) * block_size;
      sy = by * block_size;
      x1 = sx - 1;
      x2 = sx;
      y2 = block_size;
      if (y2 > image->height - sy)
         y2 = image->height - sy;
      nsc = 0;
      diff = 0;
      for (i = 0; i < block2->layer_count && block2->layers[i].q != q; ++i)
         ;
      layer2 = block2->layers + i;
      for (j = 0; j < block->layer_count; ++j)
      {
         layer1 = block->layers + j;
         if (layer1->q)
            continue;
         for (y = 0; y < y2; ++y)
         {
            c1 = image->data[(sy + y) * image->width + x1];
            c2 = image->data[(sy + y) * image->width + x2];
            if (c1 >= layer1->t && (j == block->layer_count - 1 || c1 < block->layers[j + 1].t) &&
               c2 >= layer2->t && (i == block2->layer_count - 1 || c2 < block2->layers[i + 1].t))
            {
               diff += abs(c1 - c2);
               nsc++;
            }                  
         }
         if (nsc > 0)
         {
            side_measure = (float)diff / nsc;
            if (min_side_measure < 0 || side_measure < min_side_measure)
            {
               j0 = j;
               min_side_measure = side_measure;
            }
         }
      }
   }

   block2 = blocks + (by - 1) * nx + bx;
   if (by > 0 && block2->max_plane == q)
   {
      sx = bx * block_size;
      sy = by * block_size;
      y1 = sy;
      y2 = sy - 1;
      x2 = block_size;
      if (x2 > image->width - sx)
         x2 = image->width - sx;
      nsc = 0;
      diff = 0;
      for (i = 0; i < block2->layer_count && block2->layers[i].q != q; ++i)
         ;
      layer2 = block2->layers + i;
      for (j = 0; j < block->layer_count; ++j)
      {
         layer1 = block->layers + j;
         if (layer1->q)
            continue;
         for (x = 0; x < x2; ++x)
         {
            c1 = image->data[y1 * image->width + sx + x];
            c2 = image->data[y2 * image->width + sx + x];
            if (c1 >= layer1->t && (j == block->layer_count - 1 || c1 < block->layers[j + 1].t) &&
               c2 >= layer2->t && (i == block2->layer_count - 1 || c2 < block2->layers[i + 1].t))
            {
               diff += abs(c1 - c2);
               nsc++;
            }                  
         }
         if (nsc > 0)
         {
            side_measure = (float)diff / nsc;
            if (min_side_measure < 0 || side_measure < min_side_measure)
            {
               j0 = j;
               min_side_measure = side_measure;
            }
         }
      }
   }

   block2 = blocks + (by + 1) * nx + bx;
   if (by < ny - 1 && block2->max_plane == q)
   {
      sx = bx * block_size;
      sy = (by + 1) * block_size;
      y1 = sy - 1;
      y2 = sy;
      x2 = block_size;
      if (x2 > image->width - sx)
         x2 = image->width - sx;
      nsc = 0;
      diff = 0;
      for (i = 0; i < block2->layer_count && block2->layers[i].q != q; ++i)
         ;
      layer2 = block2->layers + i;
      for (j = 0; j < block->layer_count; ++j)
      {
         layer1 = block->layers + j;
         if (layer1->q)
            continue;
         for (x = 0; x < x2; ++x)
         {
            c1 = image->data[y1 * image->width + sx + x];
            c2 = image->data[y2 * image->width + sx + x];
            if (c1 >= layer1->t && (j == block->layer_count - 1 || c1 < block->layers[j + 1].t) &&
               c2 >= layer2->t && (i == block2->layer_count - 1 || c2 < block2->layers[i + 1].t))
            {
               diff += abs(c1 - c2);
               nsc++;
            }                  
         }
         if (nsc > 0)
         {
            side_measure = (float)diff / nsc;
            if (min_side_measure < 0 || side_measure < min_side_measure)
            {
               j0 = j;
               min_side_measure = side_measure;
            }
         }
      }
   }

   if (min_side_measure >= 0 && min_side_measure < 1.5)
   {              
      block->max_plane = block->layers[j0].q = q;
      *changed = 1;
   }
}


int segment(Image **res, int* res_cnt, Image *img)
{
   int bx, by, nx, ny, cnt = 0, maxCnt = 50, i, i0, k, layer_count, q, changed;
   Layer **pool;
   Block *blocks, *block;   
   float first_mountine;

   nx = (img->width + block_size - 1) / block_size;
   ny = (img->height + block_size - 1) / block_size;

   blocks = malloc(nx * ny * sizeof(Block));

   *res_cnt = maxCnt;//nx * ny;
   *res = malloc(*res_cnt * sizeof(Image));

   layer_count = 0;
   for (by = 0; by < ny; ++by)
   {
      for (bx = 0; bx < nx; ++bx)
      {           
         block = blocks + by * nx + bx;
         block_prepare(block, img, bx, by);
         block_split(block);
         layer_count += block->layer_count;
      }
   }

   pool = (Layer**)malloc(layer_count * sizeof(Layer*));

   // fill in the pool
   for (i = 0, by = 0; by < ny; ++by)
      for (bx = 0; bx < nx; ++bx)
         for (k = 0; k < block->layer_count; ++k)
            pool[i++] = blocks[by * nx + bx].layers + k;

   first_mountine = 0;
   q = 1;
   for (k = 0; k == 0 || pool[i0]->mountine > first_mountine * mountine_ratio_treshold; ++k)
   {
      // find mountine values
      for (i = 0; i < layer_count; ++i)
         layer_calc_mountine(pool, layer_count, pool[i]);

      // find maximum
      i0 = -1;
      for (i = 0; i < layer_count; ++i)
         if (!pool[i]->q && (i0 < 0 || pool[i]->mountine > pool[i0]->mountine))
            i0 = i;
      if (i0 < 0)
         break;

      if (k == 0)
         first_mountine = pool[i0]->mountine;

      pool[i0]->q = q;

      // matching phase

      do {
         changed = 0;
         for (by = 0; by < ny; ++by)
            for (bx = 0; bx < nx; ++bx)
               block_assign_plane(img, blocks, bx, by, nx, ny, q, &changed);
      } while (changed); 

      if (cnt < maxCnt)
         plane_to_image(*res + (cnt++), blocks, img, q, nx, ny);

      q++;
   }
   //for (i = 0; i < block->layer_count; ++i)
   //   if (cnt < maxCnt)
   //      block_to_image(*res + (cnt++), block, i, img);
   *res_cnt = cnt;//nx * ny;

   // collect remaining pieces


   return 0;
}
   