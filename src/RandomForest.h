#pragma once
#include "RegressionTree.h"
#include "Bootstrap.h"

class RandomForest
{
	vector<RegressionTree> trees;
	int size;
	int endTrainingDatasetIndex;
	int endValidationDatasetIndex;

	vector<double> weightResultTrees(vector<double>& treeResults);
	vector<double> squareWeightResultTrees(vector<double>& treeResults);

public:
	RandomForest(Bootstrap b, int _size = 1, int _learningSampleSize = 2000, int _endTrainingDataset = 3900, int _endValidationDataset = 4900,
		int _minSampleSize = 10, int _maxDepthTree = 40, int _minSD = 1);

	void test(Bootstrap b);
	void testWithValidationSet(Bootstrap b);
};