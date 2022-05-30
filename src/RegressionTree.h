#pragma once
#include <random>
#include <chrono>
#include <set>
#include <queue>
#include <iomanip>
#include <algorithm>
#include "Node.h"

using namespace std;

class RegressionTree
{
private:
	int numberFeatures;
	vector<map<string, int>> attrValues;
	vector<bool> isCategorical;
	vector<string> featureNames;
	Node* root;

	vector<bool> availableFeatures;

	int minSampleSize;
	int maxDepthTree;
	int minSD;

	std::mt19937 g;

	void initializeGen();

	unsigned int getRandomNumberInt(unsigned int lowerbound, unsigned int upperbound);

	pair<double, int> findNumberExamplesAndSDcorrespondingData(int searchedValue, int attrNum, vector<pair<double, vector<int>>>& trainingData);
	double findSD(vector<pair<double, vector<int>>>& trainingData, double mean, int searchedValue, int attrNum);
	double findWeightedSD(const int attrNum, vector<pair<double, vector<int>>>& trainingData);
	vector<pair<int, double>> makeDataConcise(vector<pair<double, vector<int>>>& trainingData, int attrNum);

	double evaluateSplit(vector<pair<int, double>>& conciseData, int splitValue);
	pair<double, int> findReductionSD(vector<pair<double, vector<int>>>& trainingData, Node* node, const int attrNum);

	int findAttrMaxReductionSD(vector<pair<double, vector<int>>>& trainingData, Node* node, set<int>& usedAttr);

	vector<pair<double, vector<int>>> filterData(vector<pair<double, vector<int>>>& trainingData, int attrNum, int searchedValue);
	vector<pair<double, vector<int>>> filterDataS(vector<pair<double, vector<int>>>& trainingData, int attrNum, int splitValue);
	vector<pair<double, vector<int>>> filterDataL(vector<pair<double, vector<int>>>& trainingData, int attrNum, int splitValue);

	Node* trainModel(vector<pair<double, vector<int>>>& trainingData,
		set<int> usedAttrs = set<int>(), int currDepth = 1, Node* currParent = nullptr);
public:
	pair<double, int> findBestSplit(vector<pair<int, double>>& conciseData);

	RegressionTree();
	RegressionTree(int _numberFeatures, vector<pair<double, vector<int>>> _dataset, vector<map<string, int>> _attrValues, vector<bool> _isCategorical,
		vector<string> _featureNames, int _minSampleSize, int _maxDepthTree, int _minSD, vector<bool> _availableFeatures = vector<bool>({1,1,1,1,1,1,1,1,1,1,1}));

	void printRegressionTreeBFS();
	double predictPrice(vector<int> example);

	double getStartingSD();
};