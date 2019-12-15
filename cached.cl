/**
 * This kernel function converts a RGB image to grayscale.
 **/
__kernel void rgb2gray( __global unsigned char* inputRchannel, 
                        __global unsigned char* inputGchannel,
                        __global unsigned char* inputBchannel,
                        __global unsigned char* outputImg){
    
    int index = get_global_id(0);
    outputImg[index] = (inputRchannel[index] + inputGchannel[index] + inputBchannel[index])/3;   
}

/**
 * This kernel function convolves an image with a mask
 **/
__kernel void convolve( const unsigned int maskSize ,
                        __global unsigned char* inputImg,
                        __global float* mask,
                        __global unsigned char* outputImg){

    int gColIdx = get_global_id(0); // indice global do work-item 
    int gRowIdx = get_global_id(1);
    int lColIdx = get_local_id(0); // indice local do mesmo work-item
    int lRowIdx = get_local_id(1);
    int imgWidth = get_global_size(0);
    int imgHeight = get_global_size(1);

    int grpColIdx = get_group_id(0);
    int grpRowIdx = get_group_id(1);

    const int SUB_SIZE = 16;
    __local unsigned char cachedImg[16][16];


    // cachedImg[lRowIdx][lColIdx] = inputImg[gRowIdx*imgWidth + gColIdx];
    cachedImg[lColIdx][lRowIdx] = inputImg[gRowIdx*imgWidth + gColIdx];
    const int idx = gRowIdx*imgWidth + gColIdx;


    barrier(CLK_LOCAL_MEM_FENCE);

    if(gColIdx < maskSize/2  
        || gRowIdx < maskSize/2
        || gColIdx >= imgWidth - maskSize/2
        || gRowIdx >= imgHeight - maskSize/2){
        outputImg[idx] = 0;
    }else{

        
        int outSum = 0;

        for(int k = 0; k < maskSize; k++){
            for(int l = 0; l < maskSize; l++){

                unsigned int globalColIdx = gColIdx - maskSize/2 + k;
                unsigned int globalRowIdx = gRowIdx - maskSize/2 + l;

                unsigned int maskColIdx = maskSize-1-k;
                unsigned int maskRowIdx = maskSize-1-l;

                unsigned int maskIdx = (maskSize-1-k) + (maskSize-1-l)*maskSize;

                if (grpColIdx * SUB_SIZE > globalColIdx
                || grpRowIdx * SUB_SIZE > globalRowIdx  
                || globalColIdx - SUB_SIZE >= grpColIdx * SUB_SIZE 
                || globalRowIdx - SUB_SIZE >= grpRowIdx * SUB_SIZE) {
                    outSum += inputImg[globalRowIdx * imgWidth + globalColIdx]*mask[maskIdx];
                } else {
                    // outSum += cachedImg[globalRowIdx - grpRowIdx * SUB_SIZE][globalColIdx - grpColIdx * SUB_SIZE]*mask[maskIdx];
                    outSum += cachedImg[globalColIdx - grpColIdx * SUB_SIZE][globalRowIdx - grpRowIdx * SUB_SIZE]*mask[maskIdx];
                }

            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if(outSum < 0){
            outputImg[idx] = 0;
        } else if(outSum > 255){    
            outputImg[idx] = 255;
        } else{
            outputImg[idx] = outSum;
        }

    }
}
