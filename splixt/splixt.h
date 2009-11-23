/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

#ifndef _splixt_H_
#define _splixt_H_

// Thread block size
#define BLOCK_SIZE 16
#define MAX_LAYERS 16
#define MAX_HIST 256
#define REG_SIZE 128
#define REG_SIZE_SQ (REG_SIZE*REG_SIZE)

#define separability_treshold 0.9f
#define homogeniety_separability_treshold 0.6f
#define homogeniety_variance_treshold 11.0f
#define mountine_exp 5.4f
#define mountine_ratio_treshold 0.45f

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
   int hist[MAX_HIST];
   int layer_count;
   int rx;
   int ry; 
   int max_plane;
   float average;
   float variance;
}Region;

typedef struct tagPlane{
	float average;
	float variance;
}Plane;

#endif // _splixt_H_

