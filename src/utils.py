"""
@author: Vu Quoc Hien <NeiH4207@gmail.com>
"""

import numbers
import numpy as np
import random
import os
import torch
import shutil
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

def set_seed(seed):
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
    
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

def dtanh(x):
    return 1 / np.cosh(x) ** (0.2)

def plot_elo(ratings, save_dir):
    fig, ax = plt.subplots()
    ax.plot(ratings)
    ax.set_xlabel('Episodes (x20)')
    ax.set_ylabel('Rating')
    ax.set_title('Elo Ratings')
    ax.grid(True)
    save_path = os.path.join(save_dir, 'elo.png')
    plt.savefig(save_path)
    plt.close()


def flatten(data):
    new_data = []
    # data = copy.deepcopy(data)  
    for element in data:
        if (isinstance(element, numbers.Number) ):
            new_data.append(element)
        else:
            element = flatten(element)
            for x in element:
                new_data.append(x)
    return new_data


def soft_update(target, source, tau):
	"""
	Copies the parameters from source network (x) to target network (y) using the below update
	y = TAU*x + (1 - TAU)*y
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau
		)


def hard_update(target, source):
	"""
	Copies the parameters from source network to target network
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
	"""
	Saves the models, with all training parameters intact
	:param state:
	:param is_best:
	:param filename:
	:return:
	"""
	filename = str(episode_count) + 'checkpoint.path.rar'
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

	def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X

def plot_timeseries(history, save_dir, x_label, y_label, title):
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    save_path = os.path.join(save_dir, title + '.png')
    plt.savefig(save_path)
    plt.close()
    
def plot_timeseries(history, save_dir, x_label, y_label, title):
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    save_path = os.path.join(save_dir, title + '.png')
    plt.savefig(save_path)
    plt.close()