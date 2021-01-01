#include "simplecnn.h"
using namespace std;
using namespace cv;

extern conv_param conv_params[3];
extern fc_param fc_params[1];

float* m(Mat&);
float* cn(int, int, int, float*, conv_param&);
float* mp(int, int, int, float*);

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