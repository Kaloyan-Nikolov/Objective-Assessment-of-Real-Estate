#include "Tests.h"

vector<double> Tests::weightResultTrees(vector<double>& treeResults)
{
	double sumResults = 0;
	for (int i = 0; i < treeResults.size(); i++)
	{
		sumResults += treeResults[i];
	}
	vector<double> weights(treeResults.size(), 0);
	for (int i = 0; i < treeResults.size(); i++)
	{
		weights[i] = treeResults[i] / sumResults;
	}
	return weights;
}

vector<double> Tests::squareWeightResultTrees(vector<double>& treeResults)
{
	for (int i = 0; i < treeResults.size(); i++)
	{
		treeResults[i] = treeResults[i] * treeResults[i] * treeResults[i] * treeResults[i];
	}

	return weightResultTrees(treeResults);
}

void Tests::testWightedSD()
{
	Bootstrap b;
	RegressionTree r;
	vector<pair<int, double>> vec = { {1,20},{2,30},{3,40},{4,10},{5,50} };
	pair<double, int> res = r.findBestSplit(vec);
	cout << res.first << " " << res.second << "\n";
}

void Tests::testWeighCoefficients()
{
	vector<double> treeResults = { 0.10, 0.15 };
	vector<double> weights = weightResultTrees(treeResults);
	for (int i = 0; i < weights.size(); i++)
	{
		std::cout << weights[i] << " ";
	}
	std::cout << "\n";

	vector<double> sqWeights = squareWeightResultTrees(treeResults);
	for (int i = 0; i < sqWeights.size(); i++)
	{
		std::cout << sqWeights[i] << " ";
	}
	std::cout << "\n";
}

void Tests::testDataSD()
{
	Bootstrap b;
	b.shuffleDataset();
	std::cout << "SD of DATASET:\n";
	RegressionTree r(b.getNumberFeatures(), b.getSmallFixedSample(0, 6160), b.getAttrValues(), b.getIsCategorical(), b.getFeatureNames(), 1, 50, -5);
	std::cout << r.getStartingSD() << "\n";
}

void Tests::testNumberTreesAndAggregateFunctions()
{
	Bootstrap b;
	b.shuffleDataset();

	for (int i = 2; i <= 10; i += 2)
	{
		cout << "\n---------------------" << i << "-----------------------\n";
		RandomForest rf(b, i);
		rf.test(b);
		rf.testWithValidationSet(b);
	}

	cout << "\n---------------------" << 20 << "-----------------------\n";
	RandomForest rf1(b, 20);
	rf1.test(b);
	rf1.testWithValidationSet(b);

	cout << "\n---------------------" << 30 << "-----------------------\n";
	RandomForest rf2(b, 30);
	rf2.test(b);
	rf2.testWithValidationSet(b);

	cout << "\n---------------------" << 50 << "-----------------------\n";
	RandomForest rf3(b, 50);
	rf3.test(b);
	rf3.testWithValidationSet(b);

	cout << "\n---------------------" << 75 << "-----------------------\n";
	RandomForest rf4(b, 75);
	rf4.test(b);
	rf4.testWithValidationSet(b);

	cout << "\n---------------------" << 100 << "-----------------------\n";
	RandomForest rf5(b, 100);
	rf5.test(b);
	rf5.testWithValidationSet(b);

	cout << "\n---------------------" << 200 << "-----------------------\n";
	RandomForest rf6(b, 200);
	rf6.test(b);
	rf6.testWithValidationSet(b);
}

void Tests::testNumberTrees()
{
	Bootstrap b;
	b.shuffleDataset();

	for (int i = 2; i <= 10; i += 2)
	{
		cout << "\n---------------------" << i << "-----------------------\n";
		RandomForest rf(b, i);
		rf.testWithValidationSet(b);
	}

	cout << "\n---------------------" << 20 << "-----------------------\n";
	RandomForest rf1(b, 20);
	rf1.testWithValidationSet(b);

	cout << "\n---------------------" << 30 << "-----------------------\n";
	RandomForest rf2(b, 30);
	rf2.testWithValidationSet(b);

	cout << "\n---------------------" << 50 << "-----------------------\n";
	RandomForest rf3(b, 50);
	rf3.testWithValidationSet(b);

	cout << "\n---------------------" << 75 << "-----------------------\n";
	RandomForest rf4(b, 75);
	rf4.testWithValidationSet(b);

	cout << "\n---------------------" << 100 << "-----------------------\n";
	RandomForest rf5(b, 100);
	rf5.testWithValidationSet(b);

	cout << "\n---------------------" << 200 << "-----------------------\n";
	RandomForest rf6(b, 200);
	rf6.testWithValidationSet(b);
}

void Tests::testMinSampleSize()
{
	Bootstrap b;
	b.shuffleDataset();

	cout << "\n---------------------" << 1 << "-----------------------\n";
	RandomForest rf1(b, 30, 2000, 3900, 4900, 1);
	rf1.testWithValidationSet(b);

	cout << "\n---------------------" << 5 << "-----------------------\n";
	RandomForest rf2(b, 30, 2000, 3900, 4900, 5);
	rf2.testWithValidationSet(b);

	cout << "\n---------------------" << 10 << "-----------------------\n";
	RandomForest rf3(b, 30, 2000, 3900, 4900, 10);
	rf3.testWithValidationSet(b);

	cout << "\n---------------------" << 25 << "-----------------------\n";
	RandomForest rf4(b, 30, 2000, 3900, 4900, 25);
	rf4.testWithValidationSet(b);

	cout << "\n---------------------" << 50 << "-----------------------\n";
	RandomForest rf5(b, 30, 2000, 3900, 4900, 50);
	rf5.testWithValidationSet(b);

	cout << "\n---------------------" << 75 << "-----------------------\n";
	RandomForest rf6(b, 30, 2000, 3900, 4900, 75);
	rf6.testWithValidationSet(b);

	cout << "\n---------------------" << 100 << "-----------------------\n";
	RandomForest rf7(b, 30, 2000, 3900, 4900, 100);
	rf7.testWithValidationSet(b);

	cout << "\n---------------------" << 150 << "-----------------------\n";
	RandomForest rf8(b, 30, 2000, 3900, 4900, 150);
	rf8.testWithValidationSet(b);

	cout << "\n---------------------" << 200 << "-----------------------\n";
	RandomForest rf9(b, 30, 2000, 3900, 4900, 200);
	rf9.testWithValidationSet(b);

	cout << "\n---------------------" << 300 << "-----------------------\n";
	RandomForest rf10(b, 30, 2000, 3900, 4900, 300);
	rf10.testWithValidationSet(b);

	cout << "\n---------------------" << 500 << "-----------------------\n";
	RandomForest rf11(b, 30, 2000, 3900, 4900, 500);
	rf11.testWithValidationSet(b);

	cout << "\n---------------------" << 50 << " trees -----------------------\n";

	cout << "\n---------------------" << 1 << "-----------------------\n";
	RandomForest _rf1(b, 50, 2000, 3900, 4900, 1);
	_rf1.testWithValidationSet(b);

	cout << "\n---------------------" << 5 << "-----------------------\n";
	RandomForest _rf2(b, 50, 2000, 3900, 4900, 5);
	_rf2.testWithValidationSet(b);

	cout << "\n---------------------" << 10 << "-----------------------\n";
	RandomForest _rf3(b, 50, 2000, 3900, 4900, 10);
	_rf3.testWithValidationSet(b);

	cout << "\n---------------------" << 25 << "-----------------------\n";
	RandomForest _rf4(b, 50, 2000, 3900, 4900, 25);
	_rf4.testWithValidationSet(b);

	cout << "\n---------------------" << 50 << "-----------------------\n";
	RandomForest _rf5(b, 50, 2000, 3900, 4900, 50);
	_rf5.testWithValidationSet(b);

	cout << "\n---------------------" << 75 << "-----------------------\n";
	RandomForest _rf6(b, 50, 2000, 3900, 4900, 75);
	_rf6.testWithValidationSet(b);

	cout << "\n---------------------" << 100 << "-----------------------\n";
	RandomForest _rf7(b, 50, 2000, 3900, 4900, 100);
	_rf7.testWithValidationSet(b);

	cout << "\n---------------------" << 150 << "-----------------------\n";
	RandomForest _rf8(b, 50, 2000, 3900, 4900, 150);
	_rf8.testWithValidationSet(b);

	cout << "\n---------------------" << 200 << "-----------------------\n";
	RandomForest _rf9(b, 50, 2000, 3900, 4900, 200);
	_rf9.testWithValidationSet(b);

	cout << "\n---------------------" << 300 << "-----------------------\n";
	RandomForest _rf10(b, 50, 2000, 3900, 4900, 300);
	_rf10.testWithValidationSet(b);

	cout << "\n---------------------" << 500 << "-----------------------\n";
	RandomForest _rf11(b, 50, 2000, 3900, 4900, 500);
	_rf11.testWithValidationSet(b);
}

void Tests::testMaxDepth()
{
	Bootstrap b;
	b.shuffleDataset();

	cout << "\n---------------------" << 2 << "-----------------------\n";
	RandomForest rf1(b, 50, 2000, 3900, 4900, 10, 2);
	rf1.testWithValidationSet(b);

	cout << "\n---------------------" << 3 << "-----------------------\n";
	RandomForest rf2(b, 50, 2000, 3900, 4900, 10, 3);
	rf2.testWithValidationSet(b);

	cout << "\n---------------------" << 4 << "-----------------------\n";
	RandomForest rf3(b, 50, 2000, 3900, 4900, 10, 4);
	rf3.testWithValidationSet(b);

	cout << "\n---------------------" << 5 << "-----------------------\n";
	RandomForest rf4(b, 50, 2000, 3900, 4900, 10, 5);
	rf4.testWithValidationSet(b);

	cout << "\n---------------------" << 7 << "-----------------------\n";
	RandomForest rf5(b, 50, 2000, 3900, 4900, 10, 7);
	rf5.testWithValidationSet(b);

	cout << "\n---------------------" << 10 << "-----------------------\n";
	RandomForest rf6(b, 50, 2000, 3900, 4900, 10, 10);
	rf6.testWithValidationSet(b);

	cout << "\n---------------------" << 15 << "-----------------------\n";
	RandomForest rf7(b, 50, 2000, 3900, 4900, 10, 15);
	rf7.testWithValidationSet(b);

	cout << "\n---------------------" << 20 << "-----------------------\n";
	RandomForest rf8(b, 50, 2000, 3900, 4900, 10, 20);
	rf8.testWithValidationSet(b);

	cout << "\n---------------------" << 25 << "-----------------------\n";
	RandomForest rf9(b, 50, 2000, 3900, 4900, 10, 25);
	rf9.testWithValidationSet(b);

	cout << "\n---------------------" << 30 << "-----------------------\n";
	RandomForest rf10(b, 50, 2000, 3900, 4900, 10, 30);
	rf10.testWithValidationSet(b);

	cout << "\n---------------------" << 100 << " min sample size -----------------------\n";

	cout << "\n---------------------" << 2 << "-----------------------\n";
	RandomForest _rf1(b, 50, 2000, 3900, 4900, 100, 2);
	_rf1.testWithValidationSet(b);

	cout << "\n---------------------" << 3 << "-----------------------\n";
	RandomForest _rf2(b, 50, 2000, 3900, 4900, 100, 3);
	_rf2.testWithValidationSet(b);

	cout << "\n---------------------" << 4 << "-----------------------\n";
	RandomForest _rf3(b, 50, 2000, 3900, 4900, 100, 4);
	_rf3.testWithValidationSet(b);

	cout << "\n---------------------" << 5 << "-----------------------\n";
	RandomForest _rf4(b, 50, 2000, 3900, 4900, 100, 5);
	_rf4.testWithValidationSet(b);

	cout << "\n---------------------" << 7 << "-----------------------\n";
	RandomForest _rf5(b, 50, 2000, 3900, 4900, 100, 7);
	_rf5.testWithValidationSet(b);

	cout << "\n---------------------" << 10 << "-----------------------\n";
	RandomForest _rf6(b, 50, 2000, 3900, 4900, 100, 10);
	_rf6.testWithValidationSet(b);

	cout << "\n---------------------" << 15 << "-----------------------\n";
	RandomForest _rf7(b, 50, 2000, 3900, 4900, 100, 15);
	_rf7.testWithValidationSet(b);

	cout << "\n---------------------" << 20 << "-----------------------\n";
	RandomForest _rf8(b, 50, 2000, 3900, 4900, 100, 20);
	_rf8.testWithValidationSet(b);

	cout << "\n---------------------" << 25 << "-----------------------\n";
	RandomForest _rf9(b, 50, 2000, 3900, 4900, 100, 25);
	_rf9.testWithValidationSet(b);

	cout << "\n---------------------" << 30 << "-----------------------\n";
	RandomForest _rf10(b, 50, 2000, 3900, 4900, 100, 30);
	_rf10.testWithValidationSet(b);
}

void Tests::testMinReductionSD()
{
	Bootstrap b;
	b.shuffleDataset();

	cout << "\n---------------------" << 1 << "-----------------------\n";
	RandomForest rf1(b, 50, 2000, 3900, 4900, 50, 20, 1);
	rf1.testWithValidationSet(b);

	cout << "\n---------------------" << 2 << "-----------------------\n";
	RandomForest rf2(b, 50, 2000, 3900, 4900, 50, 20, 2);
	rf2.testWithValidationSet(b);

	cout << "\n---------------------" << 5 << "-----------------------\n";
	RandomForest rf3(b, 50, 2000, 3900, 4900, 50, 20, 5);
	rf3.testWithValidationSet(b);

	cout << "\n---------------------" << 10 << "-----------------------\n";
	RandomForest rf4(b, 50, 2000, 3900, 4900, 50, 20, 10);
	rf4.testWithValidationSet(b);

	cout << "\n---------------------" << 20 << "-----------------------\n";
	RandomForest rf5(b, 50, 2000, 3900, 4900, 50, 20, 20);
	rf5.testWithValidationSet(b);

	cout << "\n---------------------" << 50 << "-----------------------\n";
	RandomForest rf6(b, 50, 2000, 3900, 4900, 50, 20, 50);
	rf6.testWithValidationSet(b);

	cout << "\n---------------------" << 100 << "-----------------------\n";
	RandomForest rf7(b, 50, 2000, 3900, 4900, 50, 20, 100);
	rf7.testWithValidationSet(b);

	cout << "\n---------------------" << 200 << "-----------------------\n";
	RandomForest rf8(b, 50, 2000, 3900, 4900, 50, 20, 200);
	rf8.testWithValidationSet(b);

	cout << "\n---------------------" << 300 << "-----------------------\n";
	RandomForest rf9(b, 50, 2000, 3900, 4900, 50, 20, 300);
	rf9.testWithValidationSet(b);

	cout << "\n---------------------" << 500 << "-----------------------\n";
	RandomForest rf10(b, 50, 2000, 3900, 4900, 50, 20, 500);
	rf10.testWithValidationSet(b);

	cout << "\n---------------------" << 1000 << "-----------------------\n";
	RandomForest rf11(b, 50, 2000, 3900, 4900, 50, 20, 1000);
	rf11.testWithValidationSet(b);

	cout << "\n---------------------" << 2000 << "-----------------------\n";
	RandomForest rf12(b, 50, 2000, 3900, 4900, 50, 20, 2000);
	rf12.testWithValidationSet(b);
}
