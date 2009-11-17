#include "image.h"
#include "segment.h"
#include "malloc.h"
#include "memory.h"

static const int block_size = 120;
static const float separability_treshold = 0.92f;

float fsqr(float val)
{
   return val * val;
}

typedef struct Block {
   int *data;
   int *histogram;
   int bx;
   int by;
   int t0;
   int t1;
   int total_intensity;
   int object_pixel_count;
   float average_intensity;
   float cumulative_probability;
   float standard_deviation;
} Block;

int block_init (Block* block)
{
   block->data = (int*)malloc(block_size * block_size * sizeof(int));
   memset(block->data, 0xFF, block_size * block_size * sizeof(int));
   block->histogram = (int*)malloc(256 * sizeof(int));
   block->t0 = 0;
   block->t1 = 256;
   if (!block->data)
      return 1;
   return 0;
}

int block_dispose (Block* block)
{
   if (block->data != NULL)
   {
      free(block->data);
      block->data = NULL;
   }
   return 0;
}

int block_calc_props (Block* block)
{  
   int x,y,v;
   memset(block->histogram, 0, 256 * sizeof(int));

   for (y = 0; y < block_size; ++y)
      for (x = 0; x < block_size; ++x)
      {
         v = block->data[y * block_size + x];
         if (v >= 0)
            block->histogram[v]++;
      }
   
   block->total_intensity = 0;
   block->object_pixel_count = 0;
   block->standard_deviation = 0;
   for (v = 0; v < 256; ++v)
   {
      block->total_intensity += v * block->histogram[v];    
      block->object_pixel_count += block->histogram[v];
      block->standard_deviation += v * v * block->histogram[v];
   }
   block->average_intensity = ((float)block->total_intensity) / block->object_pixel_count;
   block->cumulative_probability = ((float)block->object_pixel_count) / (block_size * block_size);
   block->standard_deviation = block->standard_deviation / block->object_pixel_count - block->average_intensity * block->average_intensity;
   return 0;   
}

int block_find_treshold (Block* block)
{
   int x, y, v, t, t_best = -1;
   float max_variance = 0;
   float between_class_variance;
   int total_intensity_0, total_intensity_1, object_pixels_0, object_pixels_1;   

   total_intensity_0 = 0;
   total_intensity_1 = block->total_intensity;
   object_pixels_0 = 0;
   object_pixels_1 = block->object_pixel_count;   

   for (t = block->t0; t < block->t1; ++t)
   {               
      if (object_pixels_0 > 0 && object_pixels_1 > 0)
      {
         between_class_variance = 
            (fsqr((float)total_intensity_0 / object_pixels_0 - block->average_intensity) * object_pixels_0 + 
            fsqr((float)total_intensity_1 / object_pixels_1 - block->average_intensity) * object_pixels_1) /
            block->object_pixel_count;
         if (between_class_variance > max_variance)
         {
            max_variance = between_class_variance;
            t_best = t;
         }
      }
      total_intensity_0 += t * block->histogram[t];
      total_intensity_1 -= t * block->histogram[t];
      object_pixels_0 += block->histogram[t];
      object_pixels_1 -= block->histogram[t];
   }              

   if (max_variance < block->total_variance * separability_treshold)
      return t_best;
   return -1;
}

int block_clone (Block* dst, Block* src)
{
   *dst = *src;
   block_init(dst);
}

int block_split_list (Block* list, Block* block, Block* block1, Block* block2)
{

}  

int block_split (Block* source_block, Block* block1, Block* block2)
{
   int t = block_find_treshold(source_block);

   if (t < 0)                                                  
   {
      block1 = block2 = 0;
      return;
   }

   block_clone(block1, source_block);
   block_clone(block2, source_block);

   block1->t0 = source_block->t0;
   block1->t1 = block2->t0 = i_best;
   block2->t1 = source_block->t1;

   for (y = 0; y < block_size; ++y)
      for (x = 0; x < block_size; ++x)
      {
         v = source_block->data[y * block_size + x];
         if (v >= 0)
            if (v < i_best)
               block1->data[y * block_size + x] = v;
            else
               block2->data[y * block_size + x] = v;               
      }

   block_calc_props(block1);
   block_calc_props(block2);
   
   return 0;   
}

int block_to_image (Image* img, Block* block)
{         
   int x,y;
   image_init(img, block_size, block_size);
   
   for (y = 0; y < block_size; ++y)
      for (x = 0; x < block_size; ++x)
         img->data[y * block_size + x] = block->data[y * block_size + x];

   return 0;
}

int block_select (Block* block, Image* img, int bx, int by)
{
   int x, y, xs, ys;
   for (y = 0; y < block_size; ++y)
      for (x = 0; x < block_size; ++x)
      {
         ys = by * block_size + y;
         xs = bx*  block_size + x;
         if (xs < img->width && ys < img->height)
            block->data[y * block_size + x] = img->data[ys * img->width + xs];
         else
            block->data[y * block_size + x] = -1;
      }
   block->bx = bx;
   block->by = by;
   return 0;
}

int segment(Image **res, int* res_cnt, Image *img)
{
   int bx, by, nx, ny, cnt = 0, maxCnt = 20;
   Block block, b1, b2;
   block_init(&block);
   
   nx = (img->width + block_size - 1) / block_size;
   ny = (img->height + block_size - 1) / block_size;
   
   *res_cnt = maxCnt;//nx * ny;
   *res = malloc(*res_cnt * sizeof(Image));

   for (by = 0; by < ny; ++by)
      for (bx = 0; bx < nx; ++bx)
      {
         block_select(&block, img, bx, by);   
         block_calc_props(&block);
         block_split(&block, &b1, &b2);
         if (cnt < maxCnt)
            block_to_image(*res + (cnt++), &block);
         if (cnt < maxCnt)
            block_to_image(*res + (cnt++), &b1);
         if (cnt < maxCnt)
            block_to_image(*res + (cnt++), &b2);
      }
         
   return 0;
}
