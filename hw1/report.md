# DIP Homework 1

## 1. Program

### 1.1. Function used

* cv2 (opencv-python)
    * cv2.imread() - read image
    * cv2.imwrite() - write image
    * cv2.cvtColor() - convert image color space
    * cv2.resize() - scale image

### 1.2. Program introduction

* Implements cv2.resize() function to scale image with different methods (bilinear, bicubic).
* Embeds demo option for grading purpose.
* Environments
    * macos 10.14.6
    * python 3.7.4
        * opencv-python == 4.1.1.26

### 1.3. Program usage

* Scale a image given scaling factor and method

    ```shell
    python3 resize.py [image_path] --scale [scaling_factor] --method [scaling_method]
    ```

* Demo homework 1 with given input image path

    ```shell
    python3 resize.py [image_path] --demo
    ```

* Print help message

    ```shell
    python3 resize.py --help
    ```

<div style="page-break-after: always;"></div>

## 2. Demo

| Scale |             Bilinear             |             Bicubic             |
| :---: | :------------------------------: | :-----------------------------: |
|  1.0  |       ![](imgs/selfie.jpg)       |      ![](imgs/selfie.jpg)       |
|  0.2  | ![](imgs/selfie-linear-0.2.jpg)  | ![](imgs/selfie-cubic-0.2.jpg)  |
|  3.0  | ![](imgs/selfie-linear-3.0.jpg)  | ![](imgs/selfie-cubic-3.0.jpg)  |
| 10.0  | ![](imgs/selfie-linear-10.0.jpg) | ![](imgs/selfie-cubic-10.0.jpg) |

<div style="page-break-after: always;"></div>

## 3. Comparison

|         |                   Bilinear                   |              Bicubic              |
| :-----: | :------------------------------------------: | :-------------------------------: |
|  Pros   |                     fast                     |  smoother curves when scaling up  |
|  Cons   | artifacts on curves when scaling up, blurred |               slow                |
| Example |      ![](imgs/selfie-linear-3.0-ex.jpg)      | ![](imgs/selfie-cubic-3.0-ex.jpg) |

<div style="page-break-after: always;"></div>

## 4. Bicubic Interpolation

### 4.1. Introduction

* Bicubic interpolation uses 16 points for each target pixel to compute its value. During process, cubic functions are implemented to generate smoother result than linear functions. The following are the details.
* Our target is to modeling a function $f(x,y)$ 

$$
\begin{eqnarray} f\left( x,y \right)  & = & \sum _{ i=0 }^{ 3 }{ \sum _{ j=0 }^{ 3 }{ { a }_{ ij }{ x }^{ i }{ y }^{ j } }  }  \\  & = & \begin{bmatrix} { x }^{ 3 } & { x }^{ 2 } & { x }^{ 1 } & 1 \end{bmatrix}\begin{bmatrix} { a }_{ 33 } & { a }_{ 32 } & { a }_{ 31 } & { a }_{ 30 } \\ { a }_{ 23 } & { a }_{ 22 } & { a }_{ 21 } & { a }_{ 20 } \\ { a }_{ 13 } & { a }_{ 12 } & { a }_{ 11 } & { a }_{ 10 } \\ { a }_{ 03 } & { a }_{ 02 } & { a }_{ 01 } & { a }_{ 00 } \end{bmatrix}\begin{bmatrix} { y }^{ 3 } \\ { y }^{ 2 } \\ { y }^{ 1 } \\ 1 \end{bmatrix} \\  & = & { X }^{ T }AY \end{eqnarray}
$$

* GIven 4x4 known data points

$$
F=\begin{bmatrix} f\left( -1,-1 \right)  & f\left( -1,0 \right)  & f\left( -1,1 \right)  & f\left( -1,2 \right)  \\ f\left( 0,-1 \right)  & f\left( 0,0 \right)  & f\left( 0,1 \right)  & f\left( 0,2 \right)  \\ f\left( 1,-1 \right)  & f\left( 1,0 \right)  & f\left( 1,1 \right)  & f\left( 1,2 \right)  \\ f\left( 2,-1 \right)  & f\left( 2,0 \right)  & f\left( 2,1 \right)  & f\left( 2,2 \right)  \end{bmatrix}
$$

* We could obtain the equation

$$
\begin{eqnarray} F & = & { X }^{ T }AY \\ \begin{bmatrix} f\left( -1,-1 \right)  & f\left( -1,0 \right)  & f\left( -1,1 \right)  & f\left( -1,2 \right)  \\ f\left( 0,-1 \right)  & f\left( 0,0 \right)  & f\left( 0,1 \right)  & f\left( 0,2 \right)  \\ f\left( 1,-1 \right)  & f\left( 1,0 \right)  & f\left( 1,1 \right)  & f\left( 1,2 \right)  \\ f\left( 2,-1 \right)  & f\left( 2,0 \right)  & f\left( 2,1 \right)  & f\left( 2,2 \right)  \end{bmatrix} & = & \begin{bmatrix} { \left( -1 \right)  }^{ 3 } & { \left( -1 \right)  }^{ 2 } & { \left( -1 \right)  }^{ 1 } & { 1 } \\ 0 & 0 & 0 & 1 \\ { 1 }^{ 3 } & { 1 }^{ 2 } & { 1 }^{ 1 } & 1 \\ { 2 }^{ 3 } & { 2 }^{ 2 } & { 2 }^{ 1 } & 1 \end{bmatrix}\begin{bmatrix} { a }_{ 33 } & { a }_{ 32 } & { a }_{ 31 } & { a }_{ 30 } \\ { a }_{ 23 } & { a }_{ 22 } & { a }_{ 21 } & { a }_{ 20 } \\ { a }_{ 13 } & { a }_{ 12 } & { a }_{ 11 } & { a }_{ 10 } \\ { a }_{ 03 } & { a }_{ 02 } & { a }_{ 01 } & { a }_{ 00 } \end{bmatrix}\begin{bmatrix} { \left( -1 \right)  }^{ 3 } & 0 & { 1 }^{ 3 } & { { 2 }^{ 3 } } \\ { \left( -1 \right)  }^{ 2 } & 0 & { 1 }^{ 2 } & { 2 }^{ 2 } \\ { \left( -1 \right)  }^{ 1 } & 0 & { 1 }^{ 1 } & { 2 }^{ 1 } \\ 1 & 1 & 1 & 1 \end{bmatrix}\quad  \end{eqnarray}
$$

* Therefore, the coefficient $A$ of the model $f(x,y)$ could be obtained by calculating the inverse of $X^T$ and $Y$.

$$
A=\begin{bmatrix} -\frac { 1 }{ 6 }  & \frac { 1 }{ 2 }  & -\frac { 1 }{ 2 }  & { \frac { 1 }{ 6 }  } \\ \frac { 1 }{ 2 }  & 1 & \frac { 1 }{ 2 }  & 0 \\ -\frac { 1 }{ 3 }  & -\frac { 1 }{ 2 }  & 1 & -\frac { 1 }{ 6 }  \\ 0 & 1 & 0 & 0 \end{bmatrix}\begin{bmatrix} f\left( -1,-1 \right)  & f\left( -1,0 \right)  & f\left( -1,1 \right)  & f\left( -1,2 \right)  \\ f\left( 0,-1 \right)  & f\left( 0,0 \right)  & f\left( 0,1 \right)  & f\left( 0,2 \right)  \\ f\left( 1,-1 \right)  & f\left( 1,0 \right)  & f\left( 1,1 \right)  & f\left( 1,2 \right)  \\ f\left( 2,-1 \right)  & f\left( 2,0 \right)  & f\left( 2,1 \right)  & f\left( 2,2 \right)  \end{bmatrix}\begin{bmatrix} -\frac { 1 }{ 6 }  & \frac { 1 }{ 2 }  & -\frac { 1 }{ 3 }  & { 0 } \\ \frac { 1 }{ 2 }  & 1 & -\frac { 1 }{ 2 }  & 1 \\ -\frac { 1 }{ 2 }  & \frac { 1 }{ 2 }  & 1 & 0 \\ \frac { 1 }{ 6 }  & 0 & -\frac { 1 }{ 6 }  & 0 \end{bmatrix}
$$

* With the model $f(x,y)$, we are able to compute the data point (coordinate within $0≤x≤1$ and $0≤y≤1$) referenced from these 4x4 known data points.

### 4.2. Complexity

* Without any optimization, we could summarize the multiplication and addition steps for each pixel to compare the computational complexity between bilinear and bicubic interpolation.

|                |      Bilinear       |       Bicubic        |
| :------------: | :-----------------: | :------------------: |
|    Addition    |   $2\times2-1=3$    |   $4\times4-1=15$    |
| Multiplication | $2\times2\times2=8$ | $4\times4\times2=32$ |
|    Overall     |        $11$         |         $47$         |

* From the table above, we could conclude that bicubic interpolation is about 4 times slower than bilinear interpolation.