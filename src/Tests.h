#pragma once
#include "RandomForest.h"

class Tests
{
	static vector<double> weightResultTrees(vector<double>& treeResults);
	static vector<double> squareWeightResultTrees(vector<double>& treeResults);
public:
	static void testWightedSD();
	static void testWeighCoefficients();
	static void testDataSD();
	static void testNumberTreesAndAggregateFunctions();
	static void testNumberTrees();
	static void testMinSampleSize();
	static void testMaxDepth();
	static void testMinReductionSD();
};