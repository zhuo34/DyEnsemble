import matplotlib.pyplot as plt
import numpy as np
from filterpy.monte_carlo import systematic_resample
from DyEnsembleModel import *


class DyEnsemble:
	def __init__(self, x_dim, z_dim, t_model, *m_models):
		"""
		params:
			x_dim: dimension of state
			z_dim: dimension of measurement
			t_model: transition model(function)
			m_models: measurement models(functions)
		"""
		self.n = x_dim
		self.m = z_dim

		assert isinstance(t_model, DyEnsembleTModel)
		assert t_model.n == self.n
		self.t_model = t_model
		self.m_models = []
		for model in m_models:
			assert isinstance(model, DyEnsembleMModel)
			assert model.m == self.m
			self.m_models.append(model)
		self.n_model = len(self.m_models)

	def reset_params(self, n_particle, x_idle):
		self.n_particle = n_particle
		self.particles = np.zeros((self.n_particle, self.n))
		for i in range(self.n_particle):
			self.particles[i] = x_idle
		self.pm = np.ones(self.n_model) / self.n_model
		self.weights = np.ones(self.n_particle) / self.n_particle
		self.resampling_cnt = 0

	def __call__(self, x_idle, z, n_particle, alpha, save_pm=False):
		return self.filter(x_idle, z, n_particle, alpha, save_pm)

	def filter(self, x_idle, z, n_particle, alpha, save_pm=False):
		dataset_size = z.shape[0]
		self.x_hat = np.zeros((dataset_size, self.n))
		self.reset_params(n_particle, x_idle)
		self.alpha = alpha
		self.save_pm = save_pm
		if self.save_pm:
			self.pms = np.zeros((dataset_size, self.n_model))
		for i in range(dataset_size):
			self.x_hat[i] = self.filter_once(z[i])
			if self.save_pm:
				self.pms[i] = self.pm
		return self.x_hat.T

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
		a = self.t_model(self.particles)
		b = self.t_model.random(self.n_particle)
		self.particles = a + b

	def update(self, z):
		self.pm = self.pm ** self.alpha
		self.pm /= np.sum(self.pm)
		probs = np.zeros((self.n_model, self.n_particle))
		for i in range(self.n_model):
			z_ = self.m_models[i](self.particles)
			probs[i] = self.m_models[i].prob(z, z_)
		a = np.sum(probs * self.weights, axis=1)
		a /= np.sum(a)
		self.pm *= a
		if np.sum(self.pm) < 1e-32:
			self.pm[self.pm==0] = 1e-32
		self.pm /= np.sum(self.pm)
		self.weights *= np.average(probs, axis=0, weights=self.pm)
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

	def train(self, x, z):
		for model in self.m_models:
			model.train(x, z)
