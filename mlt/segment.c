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

typedef struct tagBlock{
   int *data;
   int *histogram;
   int bx;
   int by;
   int t0;
   int t1;
   int total_intensity;
   int object_pixel_count;
   float average_intensity;
   float overall_average_intensity;
   float cumulative_probability;
   float standard_deviation;
   float between_class_variance_item;
   struct tagBlock *next, *prev;
}Block;

Block* block_alloc ()
{
   Block* b = (Block*)malloc(sizeof(Block));
   b->data = NULL;
   b->histogram = NULL;
   return b;
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

int block_free (Block* b)
{
   block_dispose(b);
   free(b);
   return 0;
}

int block_init (Block* block)
{
   block->data = (int*)malloc(block_size * block_size * sizeof(int));
   if (!block->data)
      return 1;
   memset(block->data, 0xFF, block_size * block_size * sizeof(int));
   block->histogram = (int*)malloc(256 * sizeof(int));
   if (!block->histogram)
   {
      free(block->data);
      return 1;
   }
   block->t0 = 0;
   block->t1 = 256;
   block->next = block->prev = 0;
   return 0;
}

Block* block_clone (Block* src)
{
   Block* b = block_alloc();
   if (!b)
      return NULL;
   *b = *src;
   if (block_init(b))
   {
      block_free(b);
      return NULL;
   }
   return b;
}

int block_calc_props (Block* block)
{
   int x,y,v;
   memset(block->histogram, 0, 256 * sizeof(int));

   for (y = 0; y < block_size; ++y)
   {
      for (x = 0; x < block_size; ++x)
      {
         v = block->data[y * block_size + x];
         if (v >= 0)
            block->histogram[v]++;
      }
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

float block_calc_total_variance (Block* block)
{
   int v;
   float total_variance = 0;
   for (v = 0; v < 256; ++v)
      total_variance += fsqr(v - block->average_intensity) * block->histogram[v] / (block_size * block_size);
   return total_variance;
}

float block_calc_between_class_variance (Block* list)
{
   float between_class_variance = 0;
   while ((list = list->next) != NULL)
      between_class_variance += fsqr(list->average_intensity - list->overall_average_intensity) * list->cumulative_probability;
   return between_class_variance;
}

int block_find_treshold (Block* block)
{
   int t, t_best = -1;
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

   return t_best;
}

Block* block_split (Block* source_block)
{
   Block* new_block = NULL;
   int x, y, v, t = block_find_treshold(source_block);

   if (t < 0)
      return NULL;

   new_block = block_clone(source_block);
   if (!new_block)
      return NULL;

   new_block->t1 = source_block->t1;
   new_block->t0 = source_block->t1 = t;

   for (y = 0; y < block_size; ++y)
   {
      for (x = 0; x < block_size; ++x)
      {
         v = source_block->data[y * block_size + x];
         if (v >= t)
         {
            new_block->data[y * block_size + x] = v;
            source_block->data[y * block_size + x] = -1;
         }
      }
   }

   block_calc_props(new_block);
   block_calc_props(source_block);
   return new_block;
}

int block_sink_by_sigma (Block* head, Block* b)
{
   while (head->next != NULL && head->next->standard_deviation > b->standard_deviation)
      head = head->next;
   b->next = head->next;
   head->next = b;
   return 0;
}

Block* block_split_list (Block* block)
{
   Block *b = NULL, *c = NULL;
   Block head;
   float variance_treshold;
   head.next = block;

   block->overall_average_intensity = block->average_intensity;
   variance_treshold = block_calc_total_variance(block) * separability_treshold;

   while ((b = block_split(head.next)) != 0)
   {
      c = head.next;
      head.next = c->next;
      block_sink_by_sigma(&head, c);
      block_sink_by_sigma(&head, b);
      if (block_calc_between_class_variance(&head) > variance_treshold)
         break;
   }               

   return head.next;
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
   {
      for (x = 0; x < block_size; ++x)
      {
         ys = by * block_size + y;
         xs = bx* block_size + x;
         if (xs < img->width && ys < img->height)
            block->data[y * block_size + x] = img->data[ys * img->width + xs];
         else
            block->data[y * block_size + x] = -1;
      }
   }
   block->bx = bx;
   block->by = by;
   return 0;
}

int segment(Image **res, int* res_cnt, Image *img)
{
   int bx, by, nx, ny, cnt = 0, maxCnt = 20;
   Block *block = NULL, *list = NULL, *tmp = NULL;

   nx = (img->width + block_size - 1) / block_size;
   ny = (img->height + block_size - 1) / block_size;

   *res_cnt = maxCnt;//nx * ny;
   *res = malloc(*res_cnt * sizeof(Image));

   for (by = 0; by < ny; ++by)
   {
      for (bx = 0; bx < nx; ++bx)
      {
         block = block_alloc();
         if (block_init(block))
         {
            block_free(block);
            return 1;
         }
         block_select(block, img, bx, by);
         block_calc_props(block);
         list = block_split_list(block);
         for (;list != NULL;)
         {
            if (cnt < maxCnt)
               block_to_image(*res + (cnt++), list);
            tmp = list->next;
            block_free(list);
            list = tmp;
         }
      }
   }

   return 0;
}
