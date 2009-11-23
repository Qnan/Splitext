// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil_inline.h>
#include "../image/image.h"

#define MAX_LAYERS 16
#define MAX_HIST 256
#define REG_SIZE 128
#define BLOCK_SIZE 16

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

typedef struct tagRegion{
   Layer layers[MAX_LAYERS];
   int histogram[MAX_HIST];
   int layer_count;
   int bx;
   int by; 
   int max_plane;
   float average;
   float variance;
}Region;

typedef struct tagPlane{
	float average;
	float variance;
}Plane;


void runSplitText(int argc, char** argv)
{
   if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
      cutilDeviceInit(argc, argv);
   else
      cudaSetDevice( cutGetMaxGflopsDeviceId() );

   Image img;
   image_load(&img, "../img/test1.raw");

   // allocate host memory for matrices A and B
   unsigned int img_size = img.width * img.height;
   unsigned int mem_size = sizeof(int) * img_size;

   // allocate device memory
   int* d_img;
   Region* d_blocks;
   int nx = img.width / BLOCK_SIZE, ny = img.height / BLOCK_SIZE;
   cutilSafeCall(cudaMalloc((void**) &d_img, mem_size));
   cutilSafeCall(cudaMalloc((void**) &d_blocks, nx * sizeof(Region) * ny));
   cutilSafeCall(cudaMemcpy(d_img, img.data, mem_size, cudaMemcpyHostToDevice));

   // create and start timer
   unsigned int timer = 0;
   cutilCheckError(cutCreateTimer(&timer));
   cutilCheckError(cutStartTimer(timer));

   // setup execution parameters
   //dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
   dim3 grid(img.width / BLOCK_SIZE, img.height / BLOCK_SIZE);

   // execute the kernel
   splixt_treshold<<< grid, 1 >>>(d_img, img.width, img.height, 100);

   // check if kernel execution generated and error
   cutilCheckMsg("Kernel execution failed");

   // copy result from device to host
   cutilSafeCall(cudaMemcpy(img.data, d_img, mem_size, cudaMemcpyDeviceToHost));

   // stop and destroy timer
   cutilCheckError(cutStopTimer(timer));
   printf("Processing time: %f (ms) \n", cutGetTimerValue(timer));
   cutilCheckError(cutDeleteTimer(timer));

   // check result
   //CUTBoolean res = cutCompareL2fe(reference, h_C, size_C, 1e-6f);
   //printf("Test %s \n", (1 == res) ? "PASSED" : "FAILED");
   //if (res!=1) printDiff(reference, h_C, WC, HC);

   // save result
   image_save("../img/frag/test_bin.raw", &img);

   // clean up memory
   image_dispose(&img);
   cutilSafeCall(cudaFree(d_img));

   cudaThreadExit();
}