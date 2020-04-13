import os
import pickle
import scipy.io

# from kalman_filter import *
from particle_filter import *
from DyEnsemble import *


def load_dataset(n: int):
	neu_train_set = 'dataset/train/NeuralData' + str(n) + '.mat'
	kin_train_set = 'dataset/train/KinData' + str(n) + '.mat'
	neu_test_set = 'dataset/test/NeuralData' + str(n) + '.mat'
	kin_test_set = 'dataset/test/KinData' + str(n) + '.mat'

	neu_train = scipy.io.loadmat(neu_train_set)['NeuralData']
	kin_train = scipy.io.loadmat(kin_train_set)['KinData']
	neu_test = scipy.io.loadmat(neu_test_set)['NeuralData']
	kin_test = scipy.io.loadmat(kin_test_set)['KinData']

	return preprocess_data(kin_train, neu_train, kin_test, neu_test)


def preprocess_data(kin_train, neu_train, kin_test, neu_test):
	def smooth(x, kernel=3):
		x = np.array(x, dtype=np.float)
		y = x.copy()
		if kernel % 2 == 0:
			return x
		for j in range(x.shape[1] - kernel + 1):
			sum = x[:, j].copy()
			for k in range(kernel - 1):
				sum += x[:, j + k + 1]
			y[:, j + kernel // 2] = sum / kernel
		return y
	neu_train = smooth(neu_train, 5)
	neu_test = smooth(neu_test, 5)
	kin_train = smooth(kin_train, 5)
	kin_test = smooth(kin_test, 5)

	pos_train = kin_train
	vel_train = np.insert(np.diff(pos_train), 0, np.zeros(2), axis=1) / 100
	acc_train = np.insert(np.diff(vel_train), 0, np.zeros(2), axis=1) / 0.1

	pos_test = kin_test
	vel_test = np.insert(np.diff(pos_test), 0, np.zeros(2), axis=1) / 100
	acc_test = np.insert(np.diff(vel_test), 0, np.zeros(2), axis=1) / 0.1

	kin_train = np.concatenate((pos_train, vel_train, acc_train))
	kin_test = np.concatenate((pos_test, vel_test, acc_test))

	neu_data = np.concatenate((neu_train, neu_test))
	kin_data = np.concatenate((kin_train, kin_test))

	neu_data = scipy.stats.zscore(neu_data, axis=1, ddof=2)
	kin_data = scipy.stats.zscore(kin_data, axis=1, ddof=2)

	x_split = int(kin_data.shape[0] / 2)
	z_split = int(neu_data.shape[0] / 2)
	x_train = kin_data[:x_split].transpose()
	x_test = kin_data[x_split:].transpose()
	z_train = neu_data[:z_split].transpose()
	z_test = neu_data[z_split:].transpose()

	return x_train, z_train, x_test, z_test


def cal_mse_cc(x_hat, x_test):
	mse = np.mean((x_test - x_hat) ** 2, axis=0)
	a = x_test - x_test.mean(axis=0)
	b = x_hat - x_hat.mean(axis=0)
	cov = np.sum(a * b, axis=0)
	var = np.sqrt(np.sum(a ** 2, axis=0) * np.sum(b ** 2, axis=0))
	cc = cov / var
	return mse, cc


def test(dataset_nums, filename, n_particles):
	result = {}
	for i in range(len(dataset_nums)):
		dataset_num = dataset_nums[i]
		print('=====> do dataset', dataset_num)

		# load data
		mov_train, neu_train, mov_test, neu_test = load_dataset(dataset_num)

		mov_dim = mov_train.shape[1]
		neu_dim = neu_train.shape[1]
		mov0 = mov_test[0]

		# # Kalman Filter
		# print('=====> Kalman Filter')
		# kf = KalmanFilter(mov_dim, neu_dim)
		# kf.train(mov_train, neu_train)
		# mov_hat = kf.filter(mov0, neu_test)
		# mse, cc = cal_mse_cc(mov_hat, mov_test)
		# print('mse\t', mse)
		# print('cc\t', cc)
		# ccs_kf = cc
		
		# kalman filter with different particle numbers
		ccs_pf = {}
		pf = ParticleFilter(mov_dim, neu_dim)
		pf.train(mov_train, neu_train)
		for n in n_particles:
			print('=====> Particle Filter particles', n)
			mov_hat = pf.filter(mov0, neu_test, n)
			mse, cc = cal_mse_cc(mov_hat, mov_test)
			print('mse\t', mse)
			print('cc\t', cc)
			ccs_pf[n] = cc

		result['dataset'+str(dataset_num)] = {'KF': ccs_kf, 'PF': ccs_pf}

	# save to filenmae
	with open(filename, 'wb') as f:
		pickle.dump(result, f)


def load_results(filename):
	with open(filename, 'rb') as f:
		result = pickle.load(f)
	
	for dataset in result.keys():
		print('=====>', dataset)
		for f in result[dataset].keys():
			if f == 'KF':
				cc = result[dataset][f]
				print('=====> Kalman Filter')
				print('cc\t', cc)
			else:
				for n in result[dataset][f].keys():
					cc = result[dataset][f][n]
					print('=====> Particle Filter', n)
					print('cc\t', cc)


def main():
	# test for all datasets
	dataset_nums = list(range(1, 9))
	# test 200, 500, 1000, 2000 particles
	n_particles = [200, 500, 1000, 2000]
	test(dataset_nums, 'out/results', n_particles)


if __name__ == '__main__':
	main()
	# load_results('out/results')