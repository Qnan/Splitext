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
   int *d_image, *d_ll, *d_ref, *d_out;
   int *d_flag, flag;
   cutilSafeCall(cudaMalloc((void**) &d_image, mem_size));
   cutilSafeCall(cudaMalloc((void**) &d_out, mem_size));
   cutilSafeCall(cudaMalloc((void**) &d_ll, mem_size));   
   cutilSafeCall(cudaMalloc((void**) &d_ref, mem_size));   
   cutilSafeCall(cudaMalloc((void**) &d_flag, sizeof(int)));
   cutilSafeCall(cudaMemcpy(d_image, img.data, mem_size, cudaMemcpyHostToDevice));
   cutilSafeCall(cudaMemset(d_out, 0xFF, mem_size));

   // create and start timer
   unsigned int timer = 0;
   cutilCheckError(cutCreateTimer(&timer));
   cutilCheckError(cutStartTimer(timer));

   // binarize and search for text
   dim3 cca_threads(BLOCK_SIZE, BLOCK_SIZE);
   dim3 cca_grid(w / BLOCK_SIZE, h / BLOCK_SIZE);
   cutilCheckError(cutStartTimer(timer));

   //cust_treshold<<< cca_grid, cca_threads >>>(d_out, d_image, w, h, 100);

   if (1)
   {
      cust_cca_a_init<<< cca_grid, cca_threads >>>(d_image, w, h, d_ll);
      cutilCheckMsg("Kernel 'init' execution failed");
      while (true) {
         flag = 0;
         cutilSafeCall(cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice));
         cust_cca_a_update<<< cca_grid, cca_threads >>>(d_image, w, h, d_ll, d_flag);
         cutilCheckMsg("Kernel 'scan' execution failed");
         cutilSafeCall(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
         if (!flag)
            break;
      }
   }
   else
   {
      cust_cca_d_init<<< cca_grid, cca_threads >>>(d_image, w, h, d_ll, d_ref);
      cutilCheckMsg("Kernel 'init' execution failed");
      while (true) {
         flag = 0;
         cutilSafeCall(cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice));
         cust_cca_d_scan<<< cca_grid, cca_threads >>>(d_image, w, h, d_ll, d_ref, d_flag);
         cutilCheckMsg("Kernel 'scan' execution failed");
         cutilSafeCall(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
         if (!flag)
            break;
         cust_cca_d_resolve<<< cca_grid, cca_threads >>>(d_image, w, h, d_ll, d_ref);
         cutilCheckMsg("Kernel 'resolve' execution failed");
         cust_cca_d_relabel<<< cca_grid, cca_threads >>>(d_image, w, h, d_ll, d_ref);
         cutilCheckMsg("Kernel 'relabel' execution failed");
      }
   }

   printf("Time: %f (ms) \n", cutGetTimerValue(timer));

   // stop and destroy timer
   cutilCheckError(cutStopTimer(timer));
   //printf("Processing time: %f (ms) \n", cutGetTimerValue(timer));
   cutilCheckError(cutDeleteTimer(timer));

   // save result
   //splixt_plane_show<<< reg_grid, reg_threads >>>(d_bp, d_image, w, d_rr, nx, ny, d_pp, pc);   
   //cutilCheckMsg("Kernel execution failed");
   cutilSafeCall(cudaMemcpy(img.data, d_out, mem_size, cudaMemcpyDeviceToHost));
   image_save("../img/frag/out.raw", &img);
  
   // clean up memory
   image_dispose(&img);
   cutilSafeCall(cudaFree(d_image));
   cutilSafeCall(cudaFree(d_out));
   cutilSafeCall(cudaFree(d_ll));
   cutilSafeCall(cudaFree(d_ref));
   cutilSafeCall(cudaFree(d_flag));

   cudaThreadExit();
}

typedef struct tagLr {
   int t0, t1;
   long long v0, v1, v2;
   double avg, std, bcvc;
}Lr;

void layerCalc (Lr *lr, int* hist, int w, int h)
{
   int t, v;
   lr->v0 = lr->v1 = lr->v2 = 0;
   for (t = lr->t0; t < lr->t1; ++t)
   {
      v = hist[t];
      lr->v0 += v;
      lr->v1 += v * t;
      lr->v2 += v * t * t;
   }
   lr->avg = (double)lr->v1 / lr->v0;
   lr->std = (double)lr->v2 / lr->v0 - lr->avg * lr->avg;
   lr->bcvc = lr->std * lr->v0 / w / h;
}

int splitLayers (Lr *layers, int* hist, int w, int h)
{
   Lr *layer;
   int lc = 1;
   layers[0].t0 = 0;
   layers[0].t1 = 256;
   layerCalc(&layers[0], hist, w, h);

   double bcv_tresh = separability_treshold * layers[0].std;

   double max_sigma, max_bcv, d1, d2, bcv;
   
   int i0 = 0, i, tb, v;
   int v01, v02, v11, v12, v21, v22;
   double total_avg = layers[0].avg;
   do {  
      max_sigma = -1;
      // select layer to split
      for (i = 0; i < lc; ++i) {
         if (layers[i].bcvc > max_sigma) {
            max_sigma = layers[i].bcvc;
            i0 = i;
         }
      }      

      // find optimal treshold
      layer = &layers[i0];
      v01 = 0;
      v02 = layer->v0;
      v11 = 0;
      v12 = layer->v1;
      v21 = 0;
      v22 = layer->v2;
      
      max_bcv = 0;
      tb = -1;
      for (int t = layer->t0; t < layer->t1; ++t) {
         if (v01 > 0 && v02 > 0) {
            d1 = (double)v11 / v01 - total_avg;
            d2 = (double)v12 / v02 - total_avg;
            bcv = (d1 * d1 * v01 + d2 * d2 * v02) / layer->v0;
            if (bcv > max_bcv) {
               max_bcv = bcv;
               tb = t;
            }     
         }
         v = hist[t];
         v01 += v;
         v02 -= v;
         v *= t;
         v11 += v;
         v12 -= v;
         v *= t;
         v21 += v;
         v22 -= v;
      }
      if (tb < 0)
         break;
      for (i = lc - 1; i > i0; --i)
         layers[i + 1] = layers[i];
      lc++;
      layers[i0 + 1].t0 = tb;
      layers[i0 + 1].t1 = layers[i0].t1;
      layers[i0].t1 = tb;
      layerCalc(&layers[i0], hist, w, h);
      layerCalc(&layers[i0 + 1], hist, w, h);
   } while (max_bcv < bcv_tresh);

   return lc;
}

void runSplitText(int argc, char** argv)
{                                               
   if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
      cutilDeviceInit(argc, argv);
   else
      cudaSetDevice( cutGetMaxGflopsDeviceId() );

   // load image
   Image img;
   //image_load(&img, "../img/inn2.raw");
   image_load(&img, "../img/test.raw");
   //image_load(&img, "../img/testch.raw");
   //image_load(&img, "../img/test_big.raw");
   //image_load(&img, "../img/diehard.raw");
   //image_load(&img, "../img/manuscript.raw");
   //image_load(&img, "../img/vk.raw");
   //image_load(&img, "../img/moz.raw");
   //image_load(&img, "../img/obj.raw");
   //image_load(&img, "../img/ua.raw");
   //image_load(&img, "../img/small.raw");
   //image_load(&img, "../img/vis.raw");
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
   int *d_image;
   int *d_accu;
   int mdim = w > h ? w : h;
   ConComp *d_cc;
   ConSet *d_gg;
   int i, x, y;
   int *d_histogram, *h_histogram;
   int hsize = 265 * sizeof(int);
   h_histogram = (int*)malloc(hsize);
   cutilSafeCall(cudaMalloc((void**) &d_histogram, hsize));
   cutilSafeCall(cudaMalloc((void**) &d_image, mem_size));
   cutilSafeCall(cudaMalloc((void**) &d_bp, mem_size));   
   cutilSafeCall(cudaMalloc((void**) &d_ll, mem_size));   
   cutilSafeCall(cudaMalloc((void**) &d_ref, mem_size));   
   cutilSafeCall(cudaMalloc((void**) &d_flag, sizeof(int)));
   cutilSafeCall(cudaMalloc((void**) &d_cnt, sizeof(int)));
   cutilSafeCall(cudaMalloc((void**) &d_mnt, MAX_MNT * sizeof(Mountine)));
   cutilSafeCall(cudaMalloc((void**) &d_out, mem_size));

   cutilSafeCall(cudaMalloc((void**) &d_rr, nx * sizeof(Region) * ny));
   cutilSafeCall(cudaMalloc((void**) &d_pp, MAX_PLANES * sizeof(Plane)));
   cutilSafeCall(cudaMalloc((void**) &d_cc, MAX_CC * sizeof(ConComp)));
   cutilSafeCall(cudaMalloc((void**) &d_accu, mem_size));
   cutilSafeCall(cudaMalloc((void**) &d_gg, MAX_GG * sizeof(ConSet)));
   
   cutilSafeCall(cudaMemcpy(d_image, img.data, mem_size, cudaMemcpyHostToDevice));
   cutilSafeCall(cudaMemset(d_histogram, 0, hsize));
   cutilSafeCall(cudaMemset(d_out, 0xFF, mem_size));

   dim3 cca_threads(BLOCK_SIZE, BLOCK_SIZE);
   dim3 cca_grid(w / BLOCK_SIZE, h / BLOCK_SIZE);

   dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
   dim3 grid((w + BLOCK_SIZE - 1) / BLOCK_SIZE, (h + BLOCK_SIZE - 1) / BLOCK_SIZE);
   splixt2_calc_hist<<< grid, threads >>>(d_histogram, d_image, w, h);
   cutilSafeCall(cudaMemcpy(h_histogram, d_histogram, hsize, cudaMemcpyDeviceToHost));

   Lr layers[256];
   int lc = splitLayers(layers, h_histogram, w, h);

   Image limg;
   image_init(&limg, w, h);
   char buf[1024];
   cudaMemset(d_out, 0xFF, mem_size);
   for (i = 0; i < lc; ++i)
   {
      printf("plane %i of %i\n", i + 1, lc);
      for (y = 0; y < h; ++y)
         for (x = 0; x < w; ++x)
            if (img.data[y * w + x] >= layers[i].t0 && img.data[y * w + x] < layers[i].t1)
               limg.data[y * w + x] = img.data[y * w + x];
            else
               limg.data[y * w + x] = 0xFF;

      sprintf(buf, "../img/frag/ll_%02i.raw", i);
      image_save(buf, &limg);

      for (y = 0; y < h; ++y)
         for (x = 0; x < w; ++x)
            if (limg.data[y * w + x] != 0xFF)
               limg.data[y * w + x] = 0;

      sprintf(buf, "../img/frag/bb_%02i.raw", i);
      image_save(buf, &limg);

      cudaMemcpy(d_bp, limg.data, mem_size, cudaMemcpyHostToDevice);

      //////////////////////////////////////////////////////////
      cudaMemset(d_ll, 0xFF, mem_size);
      cust_cca_d_init<<< cca_grid, cca_threads >>>(d_bp, w, h, d_ll, d_ref);

      while (true) {
         flag = 0;
         cutilSafeCall(cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice));
         cust_cca_d_scan<<< cca_grid, cca_threads >>>(d_bp, w, h, d_ll, d_ref, d_flag);
         cutilSafeCall(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
         if (!flag)
            break;
         cust_cca_d_resolve<<< cca_grid, cca_threads >>>(d_bp, w, h, d_ll, d_ref);
         cust_cca_d_relabel<<< cca_grid, cca_threads >>>(d_bp, w, h, d_ll, d_ref);
      }      

      int ncc = 0;
      cutilSafeCall(cudaMemcpy(d_cnt, &ncc, sizeof(int), cudaMemcpyHostToDevice));
      cust_cca_collect_labels<<< cca_grid, cca_threads >>>(d_bp, w, h, d_ll, d_ref, d_cnt);
      cutilSafeCall(cudaMemcpy(&ncc, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
      if (ncc > MAX_CC)
      {
         printf("Number of connected components exceeds the maximum allowed number! Truncated!");
         ncc = MAX_CC;
      }
      if (ncc == 0)
         continue;

      cust_cca_d_relabel_1<<< cca_grid, cca_threads >>>(d_bp, w, h, d_ll, d_ref);

      int cc_blocks = (ncc + LIN_BLOCK_SIZE - 1) / LIN_BLOCK_SIZE;
      cust_cca_label_clear<<< cc_blocks, LIN_BLOCK_SIZE >>>(d_cc, ncc, w, h);

      cust_cca_labels_calc_props<<< cca_grid, cca_threads >>>(d_bp, w, h, d_ll, d_cc);

      // Y-cut
      cutilSafeCall(cudaMemset(d_accu, 0, mem_size));
      cust_xy_project_y<<< cc_blocks, LIN_BLOCK_SIZE >>>(d_cc, ncc, d_accu);
      int ng;
      cutilSafeCall(cudaMemcpy(d_cnt, &ng, sizeof(int), cudaMemcpyHostToDevice));
      cust_xy_label_accu_y<<< 1, 1 >>>(d_accu, h, d_cnt);
      cutilSafeCall(cudaMemcpy(&ng, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
      cust_xy_assign_y<<< cc_blocks, LIN_BLOCK_SIZE >>>(d_cc, ncc, d_accu, h);

      // X-cut
      cutilSafeCall(cudaMemset(d_accu, 0, mem_size));
      cust_xy_project_x<<< cc_blocks, LIN_BLOCK_SIZE >>>(d_cc, ncc, d_accu, w);
      int xlabel = 0;
      cutilSafeCall(cudaMemcpy(d_cnt, &xlabel, sizeof(int), cudaMemcpyHostToDevice));
      cust_xy_label_accu_x<<< ng, 1 >>>(d_accu, w, d_cnt);
      cutilSafeCall(cudaMemcpy(&xlabel, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
      int ncs = xlabel;
      if (ncs > MAX_GG)
         ncs = MAX_GG; // too many CS
      cust_xy_assign_x<<< cc_blocks, LIN_BLOCK_SIZE >>>(d_cc, ncc, d_accu, w);

      cust_cca_show_cc<<< cca_grid, cca_threads >>>(d_out, w, h, d_cc, ncc, d_ll);
      sprintf(buf, "../img/frag/pre_%02i.raw", i);
      cudaMemcpy(limg.data, d_out, mem_size, cudaMemcpyDeviceToHost);
      image_save(buf, &limg);

      int cs_blocks = (ncs + LIN_BLOCK_SIZE - 1) / LIN_BLOCK_SIZE;
      cust_cs_clear<<< cs_blocks, LIN_BLOCK_SIZE >>>(d_gg, ncs, w, h);
      cust_cs_props_count<<< cc_blocks, LIN_BLOCK_SIZE >>>(d_gg, ncs, d_cc, ncc);

      cust_cca_cs_is_text<<< cs_blocks, LIN_BLOCK_SIZE >>>(d_gg, ncs, d_bp, w, h);
      cust_cca_show_cs<<< cca_grid, cca_threads >>>(d_out, w, h, d_cc, d_gg, d_ll, ncs);
      //////////////////////////////////////////////////////////

      sprintf(buf, "../img/frag/out_%02i.raw", i);
      cudaMemcpy(limg.data, d_out, mem_size, cudaMemcpyDeviceToHost);
      image_save(buf, &limg);
   }
   //cudaMemcpy(limg.data, d_out, mem_size, cudaMemcpyDeviceToHost);
   //image_save("../img/frag/out.raw", &limg);
   return;

   // create and start timer
   unsigned int timer = 0;
   cutilCheckError(cutCreateTimer(&timer));
   cutilCheckError(cutStartTimer(timer));
   printf("Text time: %f (ms) \n", cutGetTimerValue(timer));
   cutilCheckError(cutStartTimer(timer));
   cutilCheckError(cutStopTimer(timer));
   cutilCheckError(cutDeleteTimer(timer));

   // save result
   //splixt_plane_show<<< reg_grid, reg_threads >>>(d_bp, d_image, w, d_rr, nx, ny, d_pp, pc);   
   //cutilCheckMsg("Kernel execution failed");
   //cutilSafeCall(cudaMemcpy(img.data, d_out, mem_size, cudaMemcpyDeviceToHost));
   //image_save("../img/frag/out.raw", &img);
  
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