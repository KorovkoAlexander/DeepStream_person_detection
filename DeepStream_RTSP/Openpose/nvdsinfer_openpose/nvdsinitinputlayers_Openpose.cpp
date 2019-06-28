/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include "nvdsinfer_custom_impl.h"

/* Assumes only one input layer "im_info" needs to be initialized */
bool NvDsInferInitializeInputLayers (std::vector<NvDsInferLayerInfo> const &inputLayersInfo,
        NvDsInferNetworkInfo const &networkInfo,
        unsigned int maxBatchSize)
{
  float *imInfo = (float *) inputLayersInfo[0].buffer;
  for (unsigned int i = 0; i < maxBatchSize; i++) {
    /* nvinfer scales input video frames to network resolution */
    imInfo[i * 3] = networkInfo.height;
    imInfo[i * 3 + 1] = networkInfo.width;
    /* The output parsing function should assume no scaling of bounding boxes is
     * required since this is done by nvinfer. */
    imInfo[i * 3 + 2] = 1;
  }

  return true;
}

