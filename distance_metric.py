from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def models(CNData, AUData):
	#suppose CNx = pixel_width, CNy = distance
	#suppose AUx = pixel_width, AUy = distance
	plt.plot(CNx, CNy)
	plt.plot(AUx, AUy)

	CNmodel = linear_model.LinearRegression()
	CNmodel.fit(CNx,CNy)

	AUmodel = linear_model.LinearRegression()
	AUmodel.fit(AUx,AUy)

	return CNmodel, AUmodel

if __name__ == "__main__":
	plt.plot(CNx, CNy)
	plt.plot(AUx, AUy)