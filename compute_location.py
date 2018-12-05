import sys
import os
from sklearn import linear_model
import numpy as np
from object_detect import CNTraining, AUTraining, TowerDetection
import geopy.distance

CN_TOWER_COORDINATES = (43.642567, -79.387054)
AURA_TOWER_COORDINATES = (43.659398, -79.382805)

def compute_distances(CoordinateSet1, CoordinateSet2):
	Distances = []
	for idx in range(len(CoordinateSet2)):
			Distances.append(geopy.distance.vincenty(CoordinateSet1, CoordinateSet2[idx]).km)
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
		d1 = np.array(dists[0]).reshape(-1,1)
		d2 = np.array(dists[1]).reshape(-1,1)
		Dist_to_CN = CNmodel.predict(d1)[0]
		Dist_to_AU = AUmodel.predict(d2)[0]
		Dist_CN_AU = 1.9011736338127032

		if(dists[2] == 0):
			lat = (Dist_CN_AU**2 + Dist_to_CN**2 - Dist_to_AU**2)/(2*Dist_CN_AU)
			lon = (Dist_to_CN**2 - lat**2)**0.5

		elif(dists[2] == 1):
			lat = (Dist_CN_AU**2 + Dist_to_CN**2 - Dist_to_AU**2)/(2*Dist_CN_AU)
			lon = -(Dist_to_CN**2 - lat**2)**0.5

		#estimated camera coordinates
		newlat = CN_TOWER_COORDINATES[0]  + (lat / 6378) * (180 / np.pi);
		newlon = CN_TOWER_COORDINATES[1]  + (lon / 6378) * (180 / np.pi);
		loc = (newlat, newlon)
		Locations.append(loc)
	return Locations

if __name__ == "__main__":
	#Get the training widths from CN Tower training images taken by the same camera
	
	CNTrainingWidths = CNTraining()
	loc_file = open("CNTrainingCoordinates.txt")
	CNTrainingLocs = loc_file.readlines()
	for idx in range(len(CNTrainingLocs)):
		CNTrainingLocs[idx] =  tuple([float(x) for x in CNTrainingLocs[idx].split()])
	Distances_to_CN = compute_distances(CN_TOWER_COORDINATES, CNTrainingLocs)

	#Get the training widths from Aura Tower training images taken by the same camera
	AUTrainingWidths = AUTraining()
	
	loc_file = open("AUTrainingCoordinates.txt")
	AUTrainingLocs = loc_file.readlines()
	for idx in range(len(AUTrainingLocs)):
		AUTrainingLocs[idx] =  tuple([float(x) for x in AUTrainingLocs[idx].split()])
	Distances_to_AU = compute_distances(AURA_TOWER_COORDINATES, AUTrainingLocs)
	
	# Create distance estimator models
	CNData = (np.array(CNTrainingWidths).reshape(-1,1), np.array(Distances_to_CN))
	AUData = (np.array(AUTrainingWidths).reshape(-1,1), np.array(Distances_to_AU))
	CNmodel, AUmodel = create_models(CNData, AUData)
	
	#Load the test image widith in pixels of detected towers
	DistanceData = TowerDetection()

	#prints coordinates of estimated location
	Locations = estimate_location(DistanceData, CNmodel, AUmodel)
	print(Locations)
