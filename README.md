# Sparse Ellipsometry: Portable Acquisition of Polarimetric SVBRDF and Shape with Unstructured Flash Photography

## Contents
1. Spatially-varying pBRDF and shape reconstruction algorithm - MATLAB
2. Model viewer - Python

## Implementation
Our codes are written in cross-platform languages, and we tested in Windows 10.
### Reconstruction algorithm
Our reconstruction algorithm runs in MATLAB R2020a (tested) with the following toolboxes:
* Computer Vision Toolbox
* Image Processing Toolbox
* Parallel Computing Toolbox
* Optimization Toolbox
* Statistics and Machine Learning Toolbox
### Viewer
Our movel viewer runs in Python 3.8 (tested) with the following libraries:
* pyopengl 3.1.1a1
* glfw 2.1.0
* numpy 1.20.2
* scipy 1.6.3

#### Installation of dependency libraries
You can set the python environment with conda and pip. First, you make the provided conda environment
```
conda env create -f environment.yml
conda activate pbrdf
```
Next, you need to install glfw.
```
pip install glfw
```


## Reconstruction algorithm
We provide a demo reconstruction code that you can run the reconstruction algorithm with a single iteration. Since the 3D reconstruction is a system software, here we provide the code implementation of our core algorithm. Besides, our system requires COLMAP, 3D data I/O, and Poisson surface reconstruction implementation additionally as follows:

* Preprocessing
  1. Calibration of camera and light positions
  2. Demosaicing of polarization images
  3. Undistortion of images
  4. COLMAP - camera position and reconstructed mesh

* Poisson surface geometry reconstruction after running our pBRDF and normal optimization algorithm through the iterative optimization framework
  1. Loading the geometry
  2. Check the visibility for each view and each vertex
  3. Poisson surface reconstruction

For the sake of reconstruction demonstration, we demonstrate the last step of iterations optimizating  geometry and pBRDF reconstruction with specular/single scattering augmentation.
In our original code implementation, our geometry reconstruction algorithm leverages Poisson surface reconstruction with the depth level of 9, resulting in ~500k vertices per object, which requires too strong computing power. 
And thus, in this demo code, we provide a precomputed intermidiate dataset with Poisson surface reconstruction with a reduced depth level of 8 for the sake of speed. 


## Model viewer
You inspect the reconstruction result in this python viewer.
Viewer is located in the ```/python/``` directory.
The default path of our viewer is ```/owl/```, so you can execute ```main_viewer.py``` directly if you unzip data file in the main directory of code.

Our viewer can render the reconstruction result from a novel view. The following instruction provides you how to chage view direction, position, and light conditions.
* Drag: rotation of the object
* ```Ctrl``` + drag: displacement of the object
* ```Shift``` + drag: rotation of the light position
* Scroll: scaling of the image
* ```Shift``` + scroll: intensity scaling of the image

You can also render Mueller matrix components with the keyboard numberpad input. The position of each numberpad key matches with the Mueller matrix component directly as follows:
```
|7 8 9|   |M_00 M_01 M_02|
|4 5 6| = |M_10 M_11 M_12|
|1 2 3|   |M_20 M_21 M_22|
```
The 0 key in the numberpad resets the viewer to the original capture setting again.   
You can also switch your view to a normal map and a rendering image. Key 1 is for the rendering image, and key 2 is for the normal image.
