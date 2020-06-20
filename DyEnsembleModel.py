import abc
import numpy as np
import scipy
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


class DyEnsembleTModel:
	def __init__(self, x_dim, step=None):
		"""
		params:
			x_dim: dimension of state
			step: variants in transition models(functions) which vary with time
		"""
		self.n = x_dim
		self.step = step

	def train(self, x):
		pass

	@abc.abstractclassmethod
	def f(self, x):
		pass

	def __call__(self, x):
		return self.f(x)

	def next_step(self):
		pass

	def random(self, size):
		return scipy.stats.multivariate_normal(np.zeros(self.n), np.identity(self.n)).rvs(size).reshape(size, self.n)


class DyEnsembleMModel:
	def __init__(self, z_dim):
		self.m = z_dim

	def train(self, x, z):
		pass

	@abc.abstractclassmethod
	def h(self, x):
		pass

	def __call__(self, x):
		return self.h(x)

	def prob(self, z, z_):
		return scipy.stats.multivariate_normal(z, np.identity(self.m)).pdf(z_)
	
	def random(self, size):
		return scipy.stats.multivariate_normal(np.zeros(self.m), np.identity(self.m)).rvs(size).reshape(size, self.m)


class DyEnsembleLinear(DyEnsembleTModel, DyEnsembleMModel):
	def __init__(self, x_dim, z_dim):
		DyEnsembleTModel.__init__(x_dim)
		DyEnsembleMModel.__init__(z_dim)

	def train(self, x, z):
		dataset_size = x.shape[0]
		x0 = x[:-1]
		x1 = x[1:]
		self.A = np.linalg.inv(x0.T @ x0) @ x0.T @ x1
		self.W = (x1 - x0 @ self.A).T @ (x1 - x0 @ self.A) / (dataset_size - 1)
		self.H = np.linalg.inv(x.T @ x) @ x.T @ z
		z_ = self.h(x)
		self.Q = (z - z_).T @ (z - z_) / dataset_size

	def f(self, x):
		return x @ self.A
	
	def random(self, size):
		i = np.arange(self.n)
		W = np.zeros((self.n, self.n))
		W[i, i] = self.W[i, i]
		w = scipy.stats.multivariate_normal(np.zeros(self.n), W).rvs(size)
		return w

	def h(self, x):
		return x @ self.H
	
	def prob(self, z, z_):
		return scipy.stats.multivariate_normal(z, self.Q).pdf(z_)


class DyEnsembleTPoly(DyEnsembleMModel):
	def __init__(self, z_dim, degree):
		super().__init__(z_dim)
		self.degree = degree
		self.poly = PolynomialFeatures(degree=self.degree)

	def train(self, x, z):
		x_ = self.poly.fit_transform(x)
		self.clf = linear_model.LinearRegression()
		self.clf.fit(x_, z)

	def h(self, x):
		x_ = self.poly.fit_transform(x)
		return self.clf.predict(x_)
