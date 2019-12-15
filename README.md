# OpenCl Convolution

This project aims to apply a low-pass and a high-pass convolutional filters to a sample image using both CPU and GPU. Whilst running on the GPU. It must use local memory as cache to accelerate the calculation process. 

The image is first converted from RGB to gray scale and is manipulated using cl::Buffer type objects. The kernels are located in separate files to ease understanding. The CImg header is also included in this repository as it is needed to render the images. 

To install the OpenCl headers on Ubuntu, run:
```
sudo apt install opencl-headers
``` 

To install the Intel OpenCl driver and the Nvidia one, run:
```
sudo apt install beignet-dev nvidia-opencl-dev
``` 

And finally, to compile and execute it, do:
```
g++ -std=c++0x -o src src.cpp -lOpenCL -lm -lpthread -ljpeg -lX11 && ./src
``` 
Where *src* is the target file.

---
Known issues:  
For some unkown reason, the cached version is only working using the intel integrated graphics. You may also need to add the -ljpeg library by yourself.  