/**
 * This kernel function converts a RGB image to grayscale.
 **/
__kernel void rgb2gray( __constant unsigned char* inputRchannel, 
                        __constant unsigned char* inputGchannel,
                        __constant unsigned char* inputBchannel,
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

    int globalColIndex = get_global_id(0); // indice global do work-item 
    int globalRowIndex = get_global_id(1);
    int colIndex = get_local_id(0); // indice local do mesmo work-item
    int rowIndex = get_local_id(1);
    int imgHeight = get_global_size(1);
    int imgWidth = get_global_size(0);

    int idx = globalRowIndex*imgWidth + globalColIndex;

    if(globalColIndex < maskSize/2  
        || globalRowIndex < maskSize/2
        || globalColIndex >= imgWidth - maskSize/2
        || globalRowIndex >= imgHeight - maskSize/2){
        outputImg[idx] = 0;
    }else{

        int outSum = 0;
        for(int k = 0; k < maskSize; k++){
            for(int l = 0; l < maskSize; l++){
                unsigned int colIdx = globalColIndex - maskSize/2 + k;
                unsigned int rowIdx = globalRowIndex - maskSize/2 + l;
                unsigned int maskIdx = (maskSize-1-k) + (maskSize-1-l)*maskSize; // pois a convolução primeiro inverte a mascara
    
            outSum += inputImg[rowIdx * imgWidth + colIdx] * mask[maskIdx];
            }
        }

        if(outSum < 0){
            outputImg[idx] = 0;
        } else if(outSum > 255){    
            outputImg[idx] = 255;
        } else{
            outputImg[idx] = outSum;
        }

    }

}
