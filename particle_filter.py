import matplotlib.pyplot as plt
import numpy as np
import scipy
from filterpy.monte_carlo import systematic_resample

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


class ParticleFilter(object):
	def __init__(self, x_dim, z_dim):
		super().__init__()
		self.n = x_dim
		self.m = z_dim

	def reset_particles(self, n_particle, x_idle):
		self.n_particle = n_particle
		self.particles = np.zeros((self.n_particle, self.n))
		for i in range(self.n_particle):
			self.particles[i] = x_idle
		self.weights = np.ones(self.n_particle)
		self.weights /= np.sum(self.weights)

	def filter(self, x_idle, z, n_particle, gui=False, x_test=None):
		dataset_size = z.shape[0]
		# z: dataset_size * self.m -> self.m * dataset_size
		# z = z.T
		self.x_hat = np.zeros((dataset_size, self.n))
		self.reset_particles(n_particle, x_idle)
		self.resampling_cnt = 0
		if gui:
			self.fig, self.ax = plt.subplots(self.n // 2, 2, sharex='col', sharey='row')
			plt.ion()
			self.epoch = 0
			self.z = z
			self.x_test = x_test
			for i in range(dataset_size):
				self.x_hat[:, i] = self.filter_once(z[:, i])
				self.particles_epoch_graph()
				self.epoch += 1
				plt.pause(0.1)
			plt.ioff()
			plt.show()
		else:
			for i in range(dataset_size):
				self.x_hat[i] = self.filter_once(z[i])
		print('resampling cnt =', self.resampling_cnt)
		return self.x_hat

	def filter_once(self, z):
		# predict
		self.predict()
		# update weights
		self.update(z)
		# resampling
		if self.neff(self.weights) < self.n_particle / 2:
			self.resampling_cnt += 1
			self.resample()
		x_hat, _ = self.estimate()
		return x_hat

	def predict(self):
		# print(np.zeros(self.n).shape, np.sqrt(np.diag(self.W)).shape)
		i = np.arange(self.n)
		W = np.zeros((self.n, self.n))
		W[i, i] = self.W[i, i]
		w1 = scipy.stats.multivariate_normal(np.zeros(self.n), W).rvs(size=self.n_particle)
		w = np.random.normal(np.zeros((self.n_particle, self.n)),
							 np.repeat(np.sqrt(np.diag(self.W)).reshape(1, self.n), self.n_particle, axis=0))
		# print(w1.shape, w.shape)
		# exit()
		self.particles = self.particles @ self.A + w1

	def update(self, z):
		if self.model == 1:
			X_ = self.poly.fit_transform(self.particles)
			z_ = self.clf.predict(X_)
		else:
			z_ = self.particles @ self.H
		prod = scipy.stats.multivariate_normal(z, self.Q).pdf(z_)
		self.weights *= prod
		self.weights /= np.sum(self.weights)

	def estimate(self):
		mean = np.average(self.particles, weights=self.weights, axis=0)
		var = np.average((self.particles - mean) ** 2, weights=self.weights, axis=0)
		return mean, var

	def resample(self):
		indexes = systematic_resample(self.weights)
		self.particles = self.particles[indexes]
		self.weights = self.weights[indexes]
		self.weights.fill(1.0 / len(self.weights))

	def neff(self, weights):
		return 1. / np.sum(np.square(weights))

	def particles_epoch_graph(self):
		epochs = np.ones(self.n_particle) * self.epoch
		color = 'g'
		alpha = .20
		if self.n_particle > 5000:
			alpha *= np.sqrt(5000) / np.sqrt(self.n_particle)
		for channel in range(self.n):
			this_plot = self.ax[channel//2, channel%2]
			this_plot.set_xlim(self.epoch-25, self.epoch)
			this_plot.scatter(epochs, np.array(self.particles[channel, :]).reshape(self.n_particle),
										alpha=alpha, color=color, marker='.')
			if self.x_test is not None:
				this_plot.scatter(self.epoch, self.x_test[channel, self.epoch],
										 marker='+', color='k', s=180, lw=3)
			this_plot.scatter(self.epoch, self.x_hat[channel, self.epoch], marker='s', color='r')
		# plt.draw()

	def train(self, x, z, model=0):
		# x: dataset_size * self.n -> self.n * dataset_size
		# z: dataset_size * self.m -> self.m * dataset_size
		# x, z = x.T, z.T
		self.model = model
		dataset_size = x.shape[0]
		x0 = x[:-1]
		x1 = x[1:]

		self.A = np.linalg.inv(x0.T @ x0) @ x0.T @ x1
		self.W = (x1 - x0 @ self.A).T @ (x1 - x0 @ self.A) / (dataset_size - 1)
		if self.model == 1:
			# z = f(x) -> f 2-degree polinomial
			# X = x.T
			# Y = z.T
			self.poly = PolynomialFeatures(degree=2)
			x_ = self.poly.fit_transform(x)
			self.clf = linear_model.LinearRegression()
			self.clf.fit(x_, z)
			z_ = self.clf.predict(x_).T
		else:
			# z = f(x) -> f linear
			self.H = np.linalg.inv(x.T @ x) @ x.T @ z
			z_ = x @ self.H
		self.Q = (z - z_).T @ (z - z_) / dataset_size

