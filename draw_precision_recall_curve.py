from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

def main():
	x = np.array([0.01, 0.06, 0.47, 0.55], dtype=np.float32)
	y = np.array([0.97, 0.90, 0.72, 0.24], dtype=np.float32)
	f = interp1d(x, y)
	f2 = interp1d(x, y, kind='cubic')
	xnew = np.linspace(0.01, 0.55, num=100, endpoint=True)
	plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
	plt.legend(['PR points', 'linearly interpolated PR curve', 'cubic'], loc='best')
	plt.show()

	# x = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.999], dtype=np.float32)
	# y1 = np.array([0.97, 0.90, 0.72, 0.24, 0.22, 0.22], dtype=np.float32)
	# y2 = np.array([0.01, 0.06, 0.47, 0.55, 0.499, 0.498], dtype=np.float32)
	# y3 = np.array([0.013, 0.12, 0.57, 0.33, 0.3, 0.31], dtype=np.float32)
	# #f = interp1d(x, y)
	# f1 = interp1d(x, y1, kind='cubic')
	# f2 = interp1d(x, y2, kind='cubic')
	# f3 = interp1d(x, y3, kind='cubic')
	# xnew = np.linspace(0.11, 0.998, num=100, endpoint=True)
	# plt.plot(xnew, f1(xnew), 'o', xnew, f2(xnew), '-', xnew, f3(xnew), '--')
	# plt.legend(['precision', 'recall', 'f1_score'], loc='best')
	# print(f3(xnew))
	# m = np.argmax(f3(xnew))
	# print('xnew[{}]: {}'.format(m, xnew[m]))
	# print('max: ', np.max(f3(xnew)))
	# print('chosen precision: ', f1(xnew[m]))
	# print('chosen recall: ', f2(xnew[m]))
	# plt.xlabel('IoU')
	# plt.ylabel('Value')
	# plt.show()

if __name__=='__main__':
	main()