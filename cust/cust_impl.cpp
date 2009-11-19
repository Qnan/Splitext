// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil_inline.h>
#include "../image/image.h"

void runTest(int argc, char** argv)
{
   if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
      cutilDeviceInit(argc, argv);
   else
      cudaSetDevice( cutGetMaxGflopsDeviceId() );

   Image img;
   image_load(&img, "../img/test.raw");

   // allocate host memory for matrices A and B
   unsigned int img_size = img.width * img.height;
   unsigned int mem_size = sizeof(int) * img_size;

   // allocate device memory
   int* d_img;
   cutilSafeCall(cudaMalloc((void**) &d_img, mem_size));

   cutilSafeCall(cudaMemcpy(d_img, img.data, mem_size, cudaMemcpyHostToDevice) );

   // create and start timer
   unsigned int timer = 0;
   cutilCheckError(cutCreateTimer(&timer));
   cutilCheckError(cutStartTimer(timer));

   // setup execution parameters
   //dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
   dim3 grid(img.width / BLOCK_SIZE, img.height / BLOCK_SIZE);

   // execute the kernel
   cust_treshold<<< grid, 1 >>>(d_img, img.width, img.height, 100);

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

void runCCA(int argc, char** argv)
{
   if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
      cutilDeviceInit(argc, argv);
   else
      cudaSetDevice( cutGetMaxGflopsDeviceId() );

   Image img;
   image_load(&img, "../img/frag/test_bin.raw");

   // allocate host memory for matrices A and B
   unsigned int img_size = img.width * img.height;
   unsigned int mem_size = sizeof(int) * img_size;

   // allocate device memory
   int *d_img, *d_ll, *d_ref;
   cutilSafeCall(cudaMalloc((void**) &d_img, mem_size));
   cutilSafeCall(cudaMalloc((void**) &d_ll, mem_size));
   cutilSafeCall(cudaMalloc((void**) &d_ref, mem_size));
   cutilSafeCall(cudaMemcpy(d_img, img.data, mem_size, cudaMemcpyHostToDevice) );

   // create and start timer
   unsigned int timer = 0;
   cutilCheckError(cutCreateTimer(&timer));
   cutilCheckError(cutStartTimer(timer));

   // setup execution parameters
   int grid_x = (img.width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
      grid_y = (img.height + BLOCK_SIZE - 1) / BLOCK_SIZE;

   dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
   dim3 grid(grid_x, grid_y);

   // execute the kernel
   cust_cca_d_init<<< grid, threads >>>(d_img, img.width, img.height, d_ll, d_ref);
   cutilCheckMsg("Kernel 'init' execution failed");

   bool flag;
   int cnt = 0;
   while (true) {
      flag = false;
      cust_cca_d_scan<<< grid, threads >>>(d_img, img.width, img.height, d_ll, d_ref, &flag);
      cutilCheckMsg("Kernel 'scan' execution failed");
      if (!flag || cnt++ > 20)
         break;
      cust_cca_d_resolve<<< grid, threads >>>(d_img, img.width, img.height, d_ll, d_ref);
      cutilCheckMsg("Kernel 'resolve' execution failed");
      cust_cca_d_relabel<<< grid, threads >>>(d_img, img.width, img.height, d_ll, d_ref);
      cutilCheckMsg("Kernel 'relabel' execution failed");
   }
   //cust_cca_d_scan<<< grid, threads >>>(d_img, img.width, img.height, d_ll, d_ref, &flag);
   //cutilCheckMsg("Kernel 'scan' execution failed");
   //cust_cca_d_relabel<<< grid, threads >>>(d_img, img.width, img.height, d_ll, d_ref);
   //cutilCheckMsg("Kernel 'relabel' execution failed");
   cust_cca_d_display<<< grid, threads >>>(d_img, img.width, img.height, d_ll);
   cutilCheckMsg("Kernel 'display' execution failed");

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
   image_save("../img/frag/test_cca.raw", &img);

   // clean up memory
   image_dispose(&img);
   cutilSafeCall(cudaFree(d_img));
   cutilSafeCall(cudaFree(d_ll));
   cutilSafeCall(cudaFree(d_ref));

   cudaThreadExit();
}
