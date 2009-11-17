#include "stdio.h"
#include "malloc.h"
#include "string.h"
#include "windows.h"
#include "image.h"
#include "segment.h"

int main (void)
{
   Image img;
   Image *res;
   int res_cnt, i;
   char buf[1024];
   image_load(&img, "../img/test.raw");

   segment(&res, &res_cnt, &img);

   for (i = 0; i < res_cnt; ++i)
   {
      sprintf(buf, "../img/frag/test_%02i.raw", i);
      image_save(buf, &res[i]);
   }
   ShellExecuteA(NULL, "open", "../ggg2png/bin/Debug/ggg2png.exe", "../img/frag/", NULL, SW_SHOWNORMAL);

   return 0;
}