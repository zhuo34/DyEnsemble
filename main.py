import os
import pickle
import scipy.io
from DyEnsemble import *
from matplotlib import pyplot as plt


class F(DyEnsembleTModel):
	def f(self, x):
		x = 1 + np.sin(0.04 * np.pi * (self.step[0] + 1)) + 0.5 * x
		self.next_step()
		return x

	def next_step(self):
		self.step[0] += 1

	def random(self, size):
		return scipy.stats.gamma(3, 2).rvs(size).reshape(size, self.n)


class H1(DyEnsembleMModel):
	def h(self, x):
		return 2 * x - 3


class H2(DyEnsembleMModel):
	def h(self, x):
		return -x + 8


class H3(DyEnsembleMModel):
	def h(self, x):
		return 0.5 * x + 5


def create_sim_data():
	X, Y = [], []
	x = 0
	f = F(1, [0])
	h1, h2, h3 = H1(1), H2(1), H3(1)
	for i in range(300):
		x = f(x) + f.random(1)
		if i < 100:
			y = h1(x) + h1.random(1)
		elif i < 200:
			y = h2(x) + h2.random(1)
		else:
			y = h3(x) + h3.random(1)
		X.append(x)
		Y.append(y)
	return np.array(X).reshape(300, 1), np.array(Y).reshape(300, 1)


if __name__ == '__main__':
	x, y = create_sim_data()
	# print(x.shape, y.shape)

	step = [0]
	x_dim, z_dim = 1, 1
	model = DyEnsemble(x_dim, z_dim, F(x_dim, step), H1(z_dim), H2(z_dim), H3(z_dim))
	model.train(x, y)

	alpha = [0.1, 0.5, 0.9]
	t = np.arange(300) + 1
	fig, axs = plt.subplots(3, len(alpha), figsize=(50, 5))
	for i in range(len(alpha)):
		model(0, y, 200, alpha[i], save_pm=True)
		plt.axis('off')
		axs[0, i].plot(t, model.pms[:, 0], 'b')
		axs[1, i].plot(t, model.pms[:, 1], 'r')
		plt.axis('on')
		axs[2, i].plot(t, model.pms[:, 2], 'g')
	plt.show()
