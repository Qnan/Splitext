#ifndef NULL
#define NULL 0
#endif

typedef struct tagImage {
   int width;
   int height;
   int xdpi;
   int ydpi;
   int* data;
} Image;

int image_init (Image* img, int width, int height);
int image_dispose (Image* img);
int image_load (Image* img, const char* filename);
int image_save (const char* filename, Image* img);
