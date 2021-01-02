# simlpecnn



## 主要功能：

* 输入一张图片后，通过简易的卷积神经网络计算得出，图片中人脸所占据的比例。



## 程序：

* ##### 头文件simplecnn.h：

* ```c++
  #pragma once
  #include<iostream>
  #include<opencv2/opencv.hpp>
  #include<math.h>
  typedef struct conv_param {
      int pad;
      int stride;
      int kernel_size;
      int in_channels;
      int out_channels;
      float* p_weight;
      float* p_bias;
  } conv_param;
  
  typedef struct fc_param {
      int in_features;
      int out_features;
      float* p_weight;
      float* p_bias;
  } fc_param;
  
  ```

* ##### face_binary_cls.cpp：

* 用于计算卷积的数据，由于代码过长不在此展示。

* ##### main.cpp:

* 函数1：

* ```c++
  float* m(Mat& img) {
  	int r = img.rows;
  	int c = img.cols;
  	float* a = new float[3 * r * c]{ 0 };
  	for (int i = 0; i < r; i++) {
  		for (int j = 0; j < c; j++) {
  			a[i * c + j] = (float)img.at<Vec3b>(i, j)[2] / (float)255;
  			a[i * c + j + r * c] = (float)img.at<Vec3b>(i, j)[1] / (float)255;
  			a[i * c + j + r * c * 2] = (float)img.at<Vec3b>(i, j)[0] / (float)255;
  		}
  	}
  	return a;
  }
  ```

* 通过opencv读取图片的BGR信息，将图片的数据转化为一维数组输出。

* 函数2：

* ```c++
  float* cn(int r, int c, int t, float* a, conv_param& p) {
  	int ic = p.in_channels;
  	int oc = p.out_channels;
  	int sk = p.kernel_size;
  	int st = p.stride;
  	if (p.pad == 1) {
  		float* b = new float[t * (r + 2) * (c + 2)]{ 0 };
  		for (int k = 0; k < t; k++) {
  			for (int i = 1; i < r + 1; i++) {
  				for (int j = 1; j < c + 1; j++) {
  					b[k * (r + 2) * (c + 2) + i * (c + 2) + j] = a[k * r * c + (i - 1) * c + j - 1];
  				}
  			}
  		}
  		delete[]a;
  		float* x = new float[oc * r * c / st / st]{ 0 };
  		int n = 0;
  		for (int k = 0; k < oc; k++) {
  			for (int i = 0; i < r ; i += st) {
  				for (int j = 0; j < c ; j += st) {
  					for (int k0 = 0; k0 < t; k0++) {
  						for (int i0 = 0; i0 < sk; i0++) {
  							for (int j0 = 0; j0 < sk; j0++) {
  								x[n] += b[k0 * (r + 2) * (c + 2) + (i + i0) * (c + 2) + j + j0]
  									* p.p_weight[k * t * sk * sk + k0 * sk * sk + i0 * sk + j0];
  							}
  						}
  					}
  					n++;
  				}
  			}
  		}
  		n = oc * r * c / st / st;
  		for (int i = 0; i < oc; i++) {
  			for (int j = 0; j < r * c / st / st; j++) {
  				x[i * r * c / st / st + j] += p.p_bias[i];
  			}
  		}
  		for (int i = 0; i < n; i++) {
  			if (x[i] < 0) {
  				x[i] = 0;
  			}
  		}
  		return x;
  	}
  }
  ```

* 用于卷积运算，在pad=1的情况下，先将原矩阵围上一圈0，然后按顺序乘以权重，得到代表运算结果的矩阵。为了便于后面的步骤，将矩阵中小于0的元素变为0。

* 函数3：

* ```c++
  float* mp(int r, int c, int t, float* a) {
  	float* x = new float[t * r * c / 4];
  	for (int k = 0; k < t; k++) {
  		for (int i = 0; i < r; i += 2) {
  			for (int j = 0; j < c; j += 2) {
  				float m = a[k * r * c + i * c + j];
  				if (m < a[k * r * c + i * c + j + 1]) {
  					m = a[k * r * c + i * c + j + 1];
  				}
  				if (m < a[k * r * c + (i + 1) * c + j]) {
  					m = a[k * r * c + (i + 1) * c + j];
  				}
  				if (m < a[k * r * c + (i + 1) * c + j + 1]) {
  					m = a[k * r * c + (i + 1) * c + j + 1];
  				}
  				x[k * r * c / 4 + i * c / 4 + j / 2] = m;
  			}
  		}
  	}
  	delete[]a;
  	return x;
  }
  ```

* 最大池化算法，将矩阵分为若干个2*2的小矩阵，在每个矩阵中取出最大值后重新组成一个新的矩阵。

* 主程序：

* ```c++
  int main() {
  	Mat img = imread("face.jpg");
  	float* a = m(img);
  	float* b = cn(128, 128, 3, a, conv_params[0]);
  	a = mp(64, 64, 16, b);
  	b = cn(32, 32, 16, a, conv_params[1]);
  	a = mp(32, 32, 32, b);
  	b = cn(16, 16, 32, a, conv_params[2]);
  	float* x = new float[2]{ 0 };
  	for (int i = 0; i < 2048; i++) {
  		x[0] += fc_params[0].p_weight[i] * b[i];
  		x[1] += fc_params[0].p_weight[i + 2048] * b[i];
  	}
  	x[0] += fc_params[0].p_bias[0];
  	x[1] += fc_params[0].p_bias[1];
  	cout << "The proportion of backgrond in the picture is:" << exp(x[0]) / (exp(x[0]) + exp(x[1])) << endl;
  	cout << "The proportion of face in the picture is:" << exp(x[1]) / (exp(x[0]) + exp(x[1])) << endl;
  	return 0;
  }
  ```

* 输入一张大小为128*128的图片，转化图片数据后，交替经过三次卷积和两次池化操作后，将得到的矩阵数据与判别人脸的权重矩阵相乘，得到图片中人脸所占比例，输出最终结果。

* ## 输出结果：

* 测试人脸时的输出结果：

* ```c++
  Mat img = imread("face.jpg");
  ```

* ![result1](https://github.com/taoge-wjcs/simlpecnn/blob/main/p/result1.png)

* 测试背景图片时的输出结果：
* ```c++
  Mat img = imread("bg.jpg");
  ```
* ![result2](https://github.com/taoge-wjcs/simlpecnn/blob/main/p/result2.png)

* 结果正确。

