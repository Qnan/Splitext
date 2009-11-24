// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil_inline.h>
#include "../image/image.h"


void runSplitText(int argc, char** argv)
{
   if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
      cutilDeviceInit(argc, argv);
   else
      cudaSetDevice( cutGetMaxGflopsDeviceId() );

   Image img;
   image_load(&img, "../img/test.raw");
   //image_load(&img, "../img/test_big.raw");

   // allocate host memory for matrices A and B
   unsigned int img_size = img.width * img.height;
   unsigned int mem_size = sizeof(int) * img_size;

   // allocate device memory
   int* d_img;
   Region* d_regs;
   Global* d_g;
   Mountine* d_mountine;
   int nx = img.width / REG_SIZE, ny = img.height / REG_SIZE;
   cutilSafeCall(cudaMalloc((void**) &d_img, mem_size));
   cutilSafeCall(cudaMemcpy(d_img, img.data, mem_size, cudaMemcpyHostToDevice));
   cutilSafeCall(cudaMalloc((void**) &d_regs, nx * sizeof(Region) * ny));
   cutilSafeCall(cudaMalloc((void**) &d_g, sizeof(Global)));
   cutilSafeCall(cudaMalloc((void**) &d_mountine, sizeof(Mountine)));

   // create and start timer
   unsigned int timer = 0;
   cutilCheckError(cutCreateTimer(&timer));
   cutilCheckError(cutStartTimer(timer));

   // setup execution parameters
   dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
   dim3 grid((nx + BLOCK_SIZE - 1) / BLOCK_SIZE, (ny + BLOCK_SIZE - 1) / BLOCK_SIZE);

   // execute the kernel
   //splixt_region_split<<< grid, 1 >>>(d_img, img.width, img.height, 100);
   splixt_region_split<<< grid, threads >>>(d_regs, nx, ny, d_img, img.width);
   splixt_plane_construct<<< grid, threads >>>(d_g, d_regs, nx, ny, d_img, img.width, d_mountine);

   // check if kernel execution generated and error
   cutilCheckMsg("Kernel execution failed");

   // copy result from device to host
   cutilSafeCall(cudaMemcpy(img.data, d_img, mem_size, cudaMemcpyDeviceToHost));

   // stop and destroy timer
   cutilCheckError(cutStopTimer(timer));
   printf("Processing time: %f (ms) \n", cutGetTimerValue(timer));
   cutilCheckError(cutDeleteTimer(timer));

   // save result
   image_save("../img/frag/test_bin.raw", &img);

   // clean up memory
   image_dispose(&img);
   cutilSafeCall(cudaFree(d_img));

   cudaThreadExit();
}