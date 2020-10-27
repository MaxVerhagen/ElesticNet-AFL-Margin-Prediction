import numpy as np
from numpy import *

import pandas as pd
from pandas import DataFrame

import scipy.stats as stats
from math import sqrt

from sklearn import model_selection

class pre_prosessing:
	def __init__(self, data):
		self.df = DataFrame(data)
		print("Input data size:",len(data))
		self.X = []
		self.Y = []


	def zscore_remove(self, zmax=6):
		z = np.abs(stats.zscore(self.df,nan_policy='omit'))
		where_are_NaNs = isnan(z)
		z[where_are_NaNs] = 0
		data_clean = (z < 6).all(axis=1)
		new_df = self.df[data_clean]
		print("Data shape after Zscore:",new_df.shape)
		self.df = new_df
		return self

	def qrange_remove(self, low=0.05, upper=0.95):
		q1 = self.df.quantile(q=0.01)
		q3 = self.df.quantile(q=0.99)
		IQR = self.df.apply(stats.iqr)
		data_clean = self.df[~((self.df < (q1-1.5*IQR)) | (self.df > (q3+1.5*IQR))).any(axis=1)]
		print("Data shape after Qrange:",data_clean.shape)
		self.df =data_clean
		return self

	def x_y_split(self):
		xylist = self.df.values.tolist()
		self.X = [item[1:] for item in xylist ]
		self.Y = [item[0] for item in xylist]
		return self

	def closegame_remove(self, min=-7, max=7):
		indexrange = len(self.Y)
		for game in range(indexrange):
		    if(game==indexrange):
		        break
		    elif(self.Y[game]<max and self.Y[game]>min):
		        self.Y.pop(game)
		        self.X.pop(game)
		        indexrange-=1
		print("Data size after close games removed:",len(self.Y))
		return self

	def data_split(self, test_size, seed, shuffle=True):
		x_train, x_test, y_train, y_test = model_selection.train_test_split(self.X, self.Y, test_size=test_size, random_state=seed, shuffle=shuffle)
		print("Traing size:",len(x_train),", Test size:",len(x_test))
		return x_train, x_test, y_train, y_test
