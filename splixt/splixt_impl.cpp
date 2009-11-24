// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil_inline.h>
#include "../image/image.h"

void filterText (int argc, char** argv)
{
}

void runSplitText(int argc, char** argv)
{
   if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
      cutilDeviceInit(argc, argv);
   else
      cudaSetDevice( cutGetMaxGflopsDeviceId() );

   // load image
   Image img;
   image_load(&img, "../img/test.raw");
   //image_load(&img, "../img/test_big.raw");
   int w = img.width, h = img.height;
   unsigned int mem_size = sizeof(int) * w * h;

   // setup execution parameters
   int nx = w / REG_SIZE;
   int ny = h / REG_SIZE;
   dim3 reg_threads(BLOCK_SIZE, BLOCK_SIZE);
   dim3 reg_grid((nx + BLOCK_SIZE - 1) / BLOCK_SIZE, (ny + BLOCK_SIZE - 1) / BLOCK_SIZE);

   // split regions into layers   
   Region* d_rr;
   int *d_ll, *d_ref, *d_bp;
   Mountine *d_mnt, h_mnt, h_mnt_ret;
   h_mnt.rx = h_mnt.ry = h_mnt.lid = -1;
   h_mnt.lock = 0;
   h_mnt.mountine = h_mnt.avg = 0.0f;
   Plane* d_pp;
   int pc;
   int *d_flag, flag;
   int *d_img;
   int i;
   cutilSafeCall(cudaMalloc((void**) &d_img, mem_size));
   cutilSafeCall(cudaMalloc((void**) &d_mnt, MAX_MNT * sizeof(Mountine)));
   cutilSafeCall(cudaMalloc((void**) &d_bp, mem_size));   
   cutilSafeCall(cudaMalloc((void**) &d_ll, mem_size));   
   cutilSafeCall(cudaMalloc((void**) &d_ref, mem_size));   
   cutilSafeCall(cudaMalloc((void**) &d_rr, nx * sizeof(Region) * ny));
   cutilSafeCall(cudaMalloc((void**) &d_pp, MAX_PLANES * sizeof(Plane)));
   cutilSafeCall(cudaMalloc((void**) &d_flag, sizeof(int)));

   cutilSafeCall(cudaMemcpy(d_img, img.data, mem_size, cudaMemcpyHostToDevice));

   // create and start timer
   unsigned int timer = 0;
   cutilCheckError(cutCreateTimer(&timer));
   cutilCheckError(cutStartTimer(timer));

   // split
   splixt_region_split<<< reg_grid, reg_threads >>>(d_rr, nx, ny, d_img, w);
   cutilCheckMsg("Kernel execution failed");

   // seed
   float first_mnt, last_mnt;
   cutilSafeCall(cudaMemcpy(d_mnt, &h_mnt, sizeof(Mountine), cudaMemcpyHostToDevice));
   splixt_calc_mountine_initial<<< reg_grid, reg_threads >>>(d_rr, nx, ny, d_mnt);
   cutilCheckMsg("Kernel execution failed");
   cutilSafeCall(cudaMemcpy(&h_mnt_ret, d_mnt, sizeof(Mountine), cudaMemcpyDeviceToHost));
   last_mnt = first_mnt = h_mnt_ret.mountine;
   for (i = 1; i < MAX_MNT && last_mnt > first_mnt * mountine_ratio_treshold; ++i)
   {  
      cutilSafeCall(cudaMemcpy(d_mnt + i, &h_mnt, sizeof(Mountine), cudaMemcpyHostToDevice));
      splixt_calc_mountine_update<<< reg_grid, reg_threads >>>(d_rr, nx, ny, d_mnt + i, d_mnt + i - 1);
      cutilCheckMsg("Kernel execution failed");
      cutilSafeCall(cudaMemcpy(&h_mnt_ret, d_mnt + i, sizeof(Mountine), cudaMemcpyDeviceToHost));
      last_mnt = h_mnt_ret.mountine;
   }
   pc = i;

   // init planes
   splixt_planes_init<<< pc, 1 >>>(d_rr, nx, ny, d_pp, d_mnt); 
   cutilCheckMsg("Kernel execution failed");

   // extend planes
   do {
      flag = 0;
      cutilSafeCall(cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice));
      splixt_plane_construct<<< reg_grid, reg_threads >>>(d_rr, nx, ny, d_pp, d_flag);
      cutilCheckMsg("Kernel execution failed");
      cutilSafeCall(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
   } while (flag);

   // binarize and search for text
   dim3 cca_threads(BLOCK_SIZE, BLOCK_SIZE);
   dim3 cca_grid(w / BLOCK_SIZE, h / BLOCK_SIZE);
   for (i = 0; i < pc; ++i)
   {
      splixt_binarize<<< reg_grid, reg_threads >>>(d_bp, d_img, w, d_rr, nx, ny, d_pp, i);
      cutilCheckMsg("Kernel execution failed");
      cust_cca_d_init<<< cca_grid, cca_threads >>>(d_bp, w, h, d_ll, d_ref);
      cutilCheckMsg("Kernel 'init' execution failed");

      while (true) {
         flag = 0;
         cutilSafeCall(cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice));
         cust_cca_d_scan<<< cca_grid, cca_threads >>>(d_bp, w, h, d_ll, d_ref, d_flag);
         cutilCheckMsg("Kernel 'scan' execution failed");
         cutilSafeCall(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
         if (!flag)
            break;
         cust_cca_d_resolve<<< cca_grid, cca_threads >>>(d_bp, w, h, d_ll, d_ref);
         cutilCheckMsg("Kernel 'resolve' execution failed");
         cust_cca_d_relabel<<< cca_grid, cca_threads >>>(d_bp, w, h, d_ll, d_ref);
         cutilCheckMsg("Kernel 'relabel' execution failed");
      }
      cust_cca_d_display<<< cca_grid, cca_threads >>>(d_bp, w, h, d_ll);
      cutilCheckMsg("Kernel execution failed");
      break;
   }

   // stop and destroy timer
   cutilCheckError(cutStopTimer(timer));
   printf("Processing time: %f (ms) \n", cutGetTimerValue(timer));
   cutilCheckError(cutDeleteTimer(timer));

   // save result
   splixt_plane_show<<< reg_grid, reg_threads >>>(d_bp, d_img, w, d_rr, nx, ny, d_pp, pc);   
   cutilCheckMsg("Kernel execution failed");
   cutilSafeCall(cudaMemcpy(img.data, d_bp, mem_size, cudaMemcpyDeviceToHost));
   image_save("../img/frag/out.raw", &img);
  
   // clean up memory
   image_dispose(&img);
   cutilSafeCall(cudaFree(d_mnt));
   cutilSafeCall(cudaFree(d_bp));
   cutilSafeCall(cudaFree(d_ll));
   cutilSafeCall(cudaFree(d_ref));
   cutilSafeCall(cudaFree(d_rr)); 
   cutilSafeCall(cudaFree(d_pp));
   cutilSafeCall(cudaFree(d_flag));

   cudaThreadExit();
}