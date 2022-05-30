#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <map>

using namespace std;

class Bootstrap
{
private:
	const int NUMBER_FEATURES = 11;
	vector<pair<double, vector<int>>> dataset; // price, features
	int datasetSize;
	vector<bool> isCategorical = { 1,0,1,1,0,0,0,0,0,1,1,0 };
	vector<map<string, int>> attrValues;
	vector<string> featureNames;
	std::mt19937 g;

	bool readInput();
	void initializeGen();
public:
	unsigned int getRandomNumberInt(unsigned int lowerbound, unsigned int upperbound);

public:

	Bootstrap();

	vector<pair<double, vector<int>>> getCompleteDataset();
	int getDatasetSize();
	vector<bool> getIsCategorical();
	vector<map<string, int>> getAttrValues();
	vector<string> getFeatureNames();
	int getNumberFeatures();

	void shuffleDataset();
	vector<pair<double, vector<int>>> getFixedSampleForTraining(int startTestDataset, int endTestDataset);
	vector<pair<double, vector<int>>> getSmallFixedSample(int start = 0, int end = 1000);

	vector<pair<double, vector<int>>> getSampleWithReplacementBefore(int desizedSize, int endTrainingDataset);
};