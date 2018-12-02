import sys
import os
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from object_detect.py import CNTraining AUTraining TowerDetection


CN_TOWER_COORDINATES = (43.6426, 79.3871)
AURA_TOWER_COORDINATES = (43.6594, 79.3828)

def compute_distances(CoordinateSet1, CoordinateSet2):
	Distances = []
	if len(CoordinateSet1) != len(CoordinateSet2):
		print("Coordinate sets differ in size: " + str(len(CoordinateSet1)) + ", " 
			+ str(len(CoordinateSet1)) + " aborting.")
		sys.exit()
	else:
		for idx in range(len(CoordinateSet1)):
			Distances.append(numpy.linalg.norm(CoordinateSet1[idx]-CoordinateSet2[idx]))
	return Distances

#given box width and distance from object, simply creates 
#linear models to estimate future distances given box width
def create_models(CNData, AUData):
	CNmodel = linear_model.LinearRegression()
	CNmodel.fit(CNData[0] , CNData[1])
	AUmodel = linear_model.LinearRegression()
	AUmodel.fit(AUData[0],AUData[1])
	return CNmodel, AUmodel

def estimate_location(DistanceData, CNmodel, AUmodel):
	Locations = []
	for dists in DistanceData:
		#Computed distances to respective locations is based on the regression models'
		#estimates of distance from the pixel widths of detected towers
		Dist_to_CN = CNmodel.predict(dists[0])
		Dist_to_AU = AUmodel.predict(dists[1])
		Dist_CN_AU = numpy.linalg.norm(CN_TOWER_COORDINATES-AURA_TOWER_COORDINATES)
		#angle between CN Tower Lattitude and CN/AURA Path
		alpha = np.sin((AURA_TOWER_COORDINATES[1]-CN_TOWER_COORDINATES[0]) / (Dist_CN_AU))
		beta = np.cos((Dist_CN_AU**2 + Dist_to_CN**2 - Dist_CN_AU**2) / _(2*Dist_CN_AU*Dist_to_CN))
		#If the CN tower is percieved to be in the left side of the image subtract alpha
		if(dists[2] == 0)
			#compute changes in lattitude and longitude relative to the CN Tower
			lat = Dist_to_CN*np.cos(alpha-beta)
			lon = Dist_to_CN*np.sin(alpha-beta)
		elif(dists[2] == 1)
			#compute changes in lattitude and longitude relative to the CN Tower
			lat = Dist_to_CN*np.cos(alpha+beta)
			lon = Dist_to_CN*np.sin(alpha+beta)
		#estimated camera coordinates
		loc = (lat, lon) + CN_TOWER_COORDINATES
		Locations.append(loc)
	return Locations

if __name__ == "__main__":
	#Get the training widths from CN Tower training images taken by the same camera
	CNTrainingWidths = CNTraining("TrainingImagesCN")
	# CNTrainingLocs = Coordinates File
	Distances_to_CN = compute_distances(CN_TOWER_COORDINATES, CNTrainingLocs)

	#Get the training widths from Aura Tower training images taken by the same camera
	AUTrainingWidths = CNTraining("TrainingImagesAura")
	
	# AUTrainingLocs = Coordinates File
	Distances_to_AU = compute_distances(AURA_TOWER_COORDINATES, AUTrainingLocs)

	# Create distance estimator models
	CNData = (CNTrainingWidths, Distances_to_CN)
	AUData = (AUTrainingWidths, Distances_to_AU)
	CNmodel, AUmodel = create_models(CNData, AUData)

	#Load the test image widith in pixels of detected towers
	DistanceData = TowerDetection("TestImages")
	Locations = estimate_location(DistanceData, CNmodel, AUmodel)
	print(Locations)

