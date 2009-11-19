#include "stdio.h"
#include "malloc.h"
#include "image.h"

int image_init (Image* img, int width, int height)
{
   img->width = width;
   img->height = height;
   img->data = (int*)malloc(width * height * sizeof(int));
   if (!img->data)
      return 1;
   return 0;
}

int image_dispose (Image* img)
{
   if (img->data != NULL)
   {
      free(img->data);
      img->data = NULL;
   }
   return 0;
}

int image_load (Image* img, const char* filename)
{
   FILE* f = fopen(filename, "rb");
   int x, y, width, height;
   fread(&width, sizeof(int), 1, f);
   fread(&height, sizeof(int), 1, f);
   image_init(img, width, height);

   fread(&img->xdpi, sizeof(int), 1, f);
   fread(&img->ydpi, sizeof(int), 1, f);
   for (y = 0; y < img->height; ++y)
      for (x = 0; x < img->width; ++x)
         fread(&img->data[y * img->width + x], sizeof(int), 1, f);
   return 0;
}

int image_save (const char* filename, Image* img)
{
   FILE* f = fopen(filename, "wb");
   fwrite(&img->width, sizeof(int), 1, f);
   fwrite(&img->height, sizeof(int), 1, f);
   fwrite(&img->xdpi, sizeof(int), 1, f);
   fwrite(&img->ydpi, sizeof(int), 1, f);
   fwrite(img->data, sizeof(int), img->width * img->height, f);
   fclose(f);
   return 0;
}
