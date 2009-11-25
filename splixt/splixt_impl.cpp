// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil_inline.h>
#include "../image/image.h"

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil_inline.h>
#include "../image/image.h"

void runCCA(int argc, char** argv)
{
   if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
      cutilDeviceInit(argc, argv);
   else
      cudaSetDevice( cutGetMaxGflopsDeviceId() );

   // load image
   Image img;
   image_load(&img, "../img/dieh100.raw");
   int w = img.width, h = img.height;
   unsigned int mem_size = sizeof(int) * w * h;

   // split regions into layers   
   int *d_img, *d_ll, *d_ref, *d_out;
   int *d_flag, flag;
   cutilSafeCall(cudaMalloc((void**) &d_img, mem_size));
   cutilSafeCall(cudaMalloc((void**) &d_out, mem_size));
   cutilSafeCall(cudaMalloc((void**) &d_ll, mem_size));   
   cutilSafeCall(cudaMalloc((void**) &d_ref, mem_size));   
   cutilSafeCall(cudaMalloc((void**) &d_flag, sizeof(int)));
   cutilSafeCall(cudaMemcpy(d_img, img.data, mem_size, cudaMemcpyHostToDevice));
   cutilSafeCall(cudaMemset(d_out, 0xFF, mem_size));

   // create and start timer
   unsigned int timer = 0;
   cutilCheckError(cutCreateTimer(&timer));
   cutilCheckError(cutStartTimer(timer));

   // binarize and search for text
   dim3 cca_threads(BLOCK_SIZE, BLOCK_SIZE);
   dim3 cca_grid(w / BLOCK_SIZE, h / BLOCK_SIZE);
   cutilCheckError(cutStartTimer(timer));

   //cust_treshold<<< cca_grid, cca_threads >>>(d_out, d_img, w, h, 100);

   if (1)
   {
      cust_cca_a_init<<< cca_grid, cca_threads >>>(d_img, w, h, d_ll);
      cutilCheckMsg("Kernel 'init' execution failed");
      while (true) {
         flag = 0;
         cutilSafeCall(cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice));
         cust_cca_a_update<<< cca_grid, cca_threads >>>(d_img, w, h, d_ll, d_flag);
         cutilCheckMsg("Kernel 'scan' execution failed");
         cutilSafeCall(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
         if (!flag)
            break;
      }
   }
   else
   {
      cust_cca_d_init<<< cca_grid, cca_threads >>>(d_img, w, h, d_ll, d_ref);
      cutilCheckMsg("Kernel 'init' execution failed");
      while (true) {
         flag = 0;
         cutilSafeCall(cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice));
         cust_cca_d_scan<<< cca_grid, cca_threads >>>(d_img, w, h, d_ll, d_ref, d_flag);
         cutilCheckMsg("Kernel 'scan' execution failed");
         cutilSafeCall(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
         if (!flag)
            break;
         cust_cca_d_resolve<<< cca_grid, cca_threads >>>(d_img, w, h, d_ll, d_ref);
         cutilCheckMsg("Kernel 'resolve' execution failed");
         cust_cca_d_relabel<<< cca_grid, cca_threads >>>(d_img, w, h, d_ll, d_ref);
         cutilCheckMsg("Kernel 'relabel' execution failed");
      }
   }

   printf("Time: %f (ms) \n", cutGetTimerValue(timer));

   // stop and destroy timer
   cutilCheckError(cutStopTimer(timer));
   //printf("Processing time: %f (ms) \n", cutGetTimerValue(timer));
   cutilCheckError(cutDeleteTimer(timer));

   // save result
   //splixt_plane_show<<< reg_grid, reg_threads >>>(d_bp, d_img, w, d_rr, nx, ny, d_pp, pc);   
   //cutilCheckMsg("Kernel execution failed");
   cutilSafeCall(cudaMemcpy(img.data, d_out, mem_size, cudaMemcpyDeviceToHost));
   image_save("../img/frag/out.raw", &img);
  
   // clean up memory
   image_dispose(&img);
   cutilSafeCall(cudaFree(d_img));
   cutilSafeCall(cudaFree(d_out));
   cutilSafeCall(cudaFree(d_ll));
   cutilSafeCall(cudaFree(d_ref));
   cutilSafeCall(cudaFree(d_flag));

   cudaThreadExit();
}

void runSplitText(int argc, char** argv)
{
   if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
      cutilDeviceInit(argc, argv);
   else
      cudaSetDevice( cutGetMaxGflopsDeviceId() );

   // load image
   Image img;
   //image_load(&img, "../img/test.raw");
   //image_load(&img, "../img/test_big.raw");
   //image_load(&img, "../img/diehard.raw");
   image_load(&img, "../img/manuscript.raw");
   int w = img.width, h = img.height;
   unsigned int mem_size = sizeof(int) * w * h;

   // setup execution parameters
   int nx = w / REG_SIZE;
   int ny = h / REG_SIZE;
   dim3 reg_threads(BLOCK_SIZE, BLOCK_SIZE);
   dim3 reg_grid((nx + BLOCK_SIZE - 1) / BLOCK_SIZE, (ny + BLOCK_SIZE - 1) / BLOCK_SIZE);

   // split regions into layers   
   Region* d_rr;
   int *d_ll, *d_ref, *d_bp, *d_out;
   Mountine *d_mnt, h_mnt, h_mnt_ret;
   h_mnt.rx = h_mnt.ry = h_mnt.lid = -1;
   h_mnt.lock = 0;
   h_mnt.mountine = h_mnt.avg = 0.0f;
   Plane* d_pp;
   int pc;
   int *d_flag, flag;
   int *d_cnt;
   int *d_img;
   int *d_accu;
   int mdim = w > h ? w : h;
   ConComp *d_cc;
   ConSet *d_gg;
   int i;
   cutilSafeCall(cudaMalloc((void**) &d_img, mem_size));
   cutilSafeCall(cudaMalloc((void**) &d_mnt, MAX_MNT * sizeof(Mountine)));
   cutilSafeCall(cudaMalloc((void**) &d_out, mem_size));
   cutilSafeCall(cudaMalloc((void**) &d_bp, mem_size));   
   cutilSafeCall(cudaMalloc((void**) &d_ll, mem_size));   
   cutilSafeCall(cudaMalloc((void**) &d_ref, mem_size));   
   cutilSafeCall(cudaMalloc((void**) &d_rr, nx * sizeof(Region) * ny));
   cutilSafeCall(cudaMalloc((void**) &d_pp, MAX_PLANES * sizeof(Plane)));
   cutilSafeCall(cudaMalloc((void**) &d_flag, sizeof(int)));
   cutilSafeCall(cudaMalloc((void**) &d_cnt, sizeof(int)));
   cutilSafeCall(cudaMalloc((void**) &d_cc, MAX_CC * sizeof(ConComp)));
   cutilSafeCall(cudaMalloc((void**) &d_accu, mem_size));
   cutilSafeCall(cudaMalloc((void**) &d_gg, MAX_GG * sizeof(ConSet)));
   
   cutilSafeCall(cudaMemcpy(d_img, img.data, mem_size, cudaMemcpyHostToDevice));

   cutilSafeCall(cudaMemset(d_out, 0xFF, mem_size));

   // create and start timer
   unsigned int timer = 0;
   cutilCheckError(cutCreateTimer(&timer));
   cutilCheckError(cutStartTimer(timer));

   // split
   splixt_region_split<<< reg_grid, reg_threads >>>(d_rr, nx, ny, d_img, w);
   cutilCheckMsg("Kernel execution failed");

   printf("Split time: %f (ms) \n", cutGetTimerValue(timer));
   cutilCheckError(cutStartTimer(timer));

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

   printf("Plane construction time: %f (ms) \n", cutGetTimerValue(timer));
   cutilCheckError(cutStartTimer(timer));

   // binarize and search for text
   dim3 cca_threads(BLOCK_SIZE, BLOCK_SIZE);
   dim3 cca_grid(w / BLOCK_SIZE, h / BLOCK_SIZE);
   int ncc = 0;
   char buf[1024];
   for (i = 0; i < pc && i < 50; ++i)
   {
      cutilCheckError(cutStartTimer(timer));

      splixt_binarize<<< reg_grid, reg_threads >>>(d_bp, d_img, w, d_rr, nx, ny, d_pp, i);
      cutilCheckMsg("Kernel execution failed");

      cutilSafeCall(cudaMemcpy(img.data, d_bp, mem_size, cudaMemcpyDeviceToHost));
      sprintf(buf, "../img/frag/big_%02i.raw", i);
      image_save(buf, &img);
      //continue;
      //image_save("../img/frag/big_bin.raw", &img);

      printf("Binarization time: %f (ms) \n", cutGetTimerValue(timer));
      cutilCheckError(cutStartTimer(timer));

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
      printf("CCA time: %f (ms) \n", cutGetTimerValue(timer));
      cutilCheckError(cutStartTimer(timer));

      ncc = 0;
      cutilSafeCall(cudaMemcpy(d_cnt, &ncc, sizeof(int), cudaMemcpyHostToDevice));
      cust_cca_collect_labels<<< cca_grid, cca_threads >>>(d_bp, w, h, d_ll, d_ref, d_cnt);
      cutilCheckMsg("Kernel execution failed");
      cutilSafeCall(cudaMemcpy(&ncc, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
      if (ncc > MAX_CC)
      {
         printf("Number of connected components exceeds the maximum allowed number! Truncated!");
         ncc = MAX_CC;
      }
      if (ncc == 0)
         continue;

      cust_cca_d_relabel_1<<< cca_grid, cca_threads >>>(d_bp, w, h, d_ll, d_ref);
      cutilCheckMsg("Kernel execution failed");

      int cc_blocks = (ncc + LIN_BLOCK_SIZE - 1) / LIN_BLOCK_SIZE;
      cust_cca_label_clear<<< cc_blocks, LIN_BLOCK_SIZE >>>(d_cc, ncc, w, h);
      cutilCheckMsg("Kernel execution failed");

      cust_cca_labels_calc_props<<< cca_grid, cca_threads >>>(d_bp, w, h, d_ll, d_cc);
      cutilCheckMsg("Kernel execution failed");

      printf("Relabeling time: %f (ms) \n", cutGetTimerValue(timer));
      cutilCheckError(cutStartTimer(timer));

      // Y-cut
      cutilSafeCall(cudaMemset(d_accu, 0, mem_size));
      cust_xy_project_y<<< cc_blocks, LIN_BLOCK_SIZE >>>(d_cc, ncc, d_accu);
      cutilCheckMsg("Kernel execution failed");

      int ng;
      cutilSafeCall(cudaMemcpy(d_cnt, &ng, sizeof(int), cudaMemcpyHostToDevice));
      cust_xy_label_accu_y<<< 1, 1 >>>(d_accu, h, d_cnt);
      cutilCheckMsg("Kernel execution failed");
      cutilSafeCall(cudaMemcpy(&ng, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));

      cust_xy_assign_y<<< cc_blocks, LIN_BLOCK_SIZE >>>(d_cc, ncc, d_accu, h);
      cutilCheckMsg("Kernel execution failed");

      // X-cut
      cutilSafeCall(cudaMemset(d_accu, 0, mem_size));
      cust_xy_project_x<<< cc_blocks, LIN_BLOCK_SIZE >>>(d_cc, ncc, d_accu, w);
      cutilCheckMsg("Kernel execution failed");

      int xlabel = 0;
      cutilSafeCall(cudaMemcpy(d_cnt, &xlabel, sizeof(int), cudaMemcpyHostToDevice));
      cust_xy_label_accu_x<<< ng, 1 >>>(d_accu, w, d_cnt);
      cutilCheckMsg("Kernel execution failed");
      cutilSafeCall(cudaMemcpy(&xlabel, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
      int ncs = xlabel+1;
      if (ncs > MAX_GG)
         ncs = MAX_GG; // too many CS
      cust_xy_assign_x<<< cc_blocks, LIN_BLOCK_SIZE >>>(d_cc, ncc, d_accu, w);
      cutilCheckMsg("Kernel execution failed");

      //cust_cca_show_cc<<< cca_grid, cca_threads >>>(d_bp, w, h, d_cc, ncc, d_ll);

      int cs_blocks = (ncs + LIN_BLOCK_SIZE - 1) / LIN_BLOCK_SIZE;
      cust_cs_clear<<< cs_blocks, LIN_BLOCK_SIZE >>>(d_gg, ncs, w, h);
      cutilCheckMsg("Kernel execution failed");
      cust_cs_props_count<<< cc_blocks, LIN_BLOCK_SIZE >>>(d_gg, ncs, d_cc, ncc);
      cutilCheckMsg("Kernel execution failed");

      printf("XY-cut time: %f (ms) \n", cutGetTimerValue(timer));
      cutilCheckError(cutStartTimer(timer));

      cust_cca_cs_is_text<<< cs_blocks, LIN_BLOCK_SIZE >>>(d_gg, ncs, d_bp, w, h);
      cust_cca_show_cs<<< cca_grid, cca_threads >>>(d_out, w, h, d_cc, d_gg, d_ll);
   }
   printf("Text time: %f (ms) \n", cutGetTimerValue(timer));
   cutilCheckError(cutStartTimer(timer));

   // stop and destroy timer
   cutilCheckError(cutStopTimer(timer));
   //printf("Processing time: %f (ms) \n", cutGetTimerValue(timer));
   cutilCheckError(cutDeleteTimer(timer));

   // save result
   //splixt_plane_show<<< reg_grid, reg_threads >>>(d_bp, d_img, w, d_rr, nx, ny, d_pp, pc);   
   //cutilCheckMsg("Kernel execution failed");
   cutilSafeCall(cudaMemcpy(img.data, d_out, mem_size, cudaMemcpyDeviceToHost));
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
   cutilSafeCall(cudaFree(d_cnt));

   cudaThreadExit();
}