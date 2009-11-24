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
   //image_load(&img, "../img/test.raw");
   image_load(&img, "../img/test_big.raw");

   // allocate host memory for matrices A and B
   unsigned int img_size = img.width * img.height;
   unsigned int mem_size = sizeof(int) * img_size;
   
   // allocate device memory
   Global *d_g;
   cutilSafeCall(cudaMalloc((void**) &d_g, sizeof(Global)));
   d_g->nx = img.width / REG_SIZE, d_g->ny = img.height / REG_SIZE;
   d_g->w = img.width;
   d_g->h = img.height;

   cutilSafeCall(cudaMalloc((void**) &d_g->img, mem_size));
   cutilSafeCall(cudaMemcpy(d_g->img, img.data, mem_size, cudaMemcpyHostToDevice));
   cutilSafeCall(cudaMalloc((void**) &d_g->out_img, mem_size));
   cutilSafeCall(cudaMalloc((void**) &d_g->rr, d_g->nx * sizeof(Region) * d_g->ny));

   Mountine *d_mnt, h_mnt;
   h_mnt.rx = h_mnt.ry = h_mnt.lid = -1;
   h_mnt.lock = 0;
   h_mnt.mountine = h_mnt.avg = 0.0f;

   cutilSafeCall(cudaMalloc((void**) &d_mnt, MAX_MNT * sizeof(Mountine)));

   // create and start timer
   unsigned int timer = 0;
   cutilCheckError(cutCreateTimer(&timer));
   cutilCheckError(cutStartTimer(timer));

   // setup execution parameters
   dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
   dim3 grid((d_g->nx + BLOCK_SIZE - 1) / BLOCK_SIZE, (d_g->ny + BLOCK_SIZE - 1) / BLOCK_SIZE);

   // split regions into layers
   splixt_region_split<<< grid, threads >>>(d_g);

   // find seeding layers
   float first_mnt, last_mnt;
   int i;
   for (i = 0; i < MAX_MNT; ++i)
   {  
      cutilSafeCall(cudaMemcpy(&d_mnt[i], &h_mnt, sizeof(Mountine), cudaMemcpyHostToDevice));
      if (i == 0)
      {
         splixt_calc_mountine_initial<<< grid, threads >>>(d_g, &d_mnt[0]);
         cutilSafeCall(cudaMemcpy(&first_mnt, &d_mnt[0].mountine, sizeof(float), cudaMemcpyDeviceToHost));
      }
      else                                                                 
      {
         splixt_calc_mountine_update<<< grid, threads >>>(d_g, &d_mnt[i], &d_mnt[i - 1]);
         cutilSafeCall(cudaMemcpy(&last_mnt, &d_mnt[i].mountine, sizeof(float), cudaMemcpyDeviceToHost));
         if (last_mnt < first_mnt * mountine_ratio_treshold)
            break;
      }
   }
   splixt_out_img_clear<<< grid, threads >>>(d_g);   
   //splixt_seed_show<<< grid, threads >>>(d_g, d_mnt, i);   
   splixt_planes_init<<< grid, threads >>>(d_g, d_mnt, i);   

   // check if kernel execution generated and error
   cutilCheckMsg("Kernel execution failed");

   // copy result from device to host
   cutilSafeCall(cudaMemcpy(img.data, d_g->out_img, mem_size, cudaMemcpyDeviceToHost));

   // stop and destroy timer
   cutilCheckError(cutStopTimer(timer));
   printf("Processing time: %f (ms) \n", cutGetTimerValue(timer));
   cutilCheckError(cutDeleteTimer(timer));

   // save result
   image_save("../img/frag/out.raw", &img);

   // clean up memory
   image_dispose(&img);
   cutilSafeCall(cudaFree(d_g->img));
   cutilSafeCall(cudaFree(d_g->rr));
   cutilSafeCall(cudaFree(d_g->pp));
   cutilSafeCall(cudaFree(d_g));

   cudaThreadExit();
}