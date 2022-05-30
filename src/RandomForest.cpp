#include "RandomForest.h"

vector<double> RandomForest::weightResultTrees(vector<double>& treeResults)
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

vector<double> RandomForest::squareWeightResultTrees(vector<double>& treeResults)
{
	for (int i = 0; i < treeResults.size(); i++)
	{
		treeResults[i] = treeResults[i] * treeResults[i] * treeResults[i] * treeResults[i];
	}
	return weightResultTrees(treeResults);
}

RandomForest::RandomForest(Bootstrap b, int _size, int _learningSampleSize, int _endTrainingDataset, int _endValidationDataset, 
	int _minSampleSize, int _maxDepthTree, int _minSD)
{
	size = _size;
	endTrainingDatasetIndex = _endTrainingDataset;
	endValidationDatasetIndex = _endValidationDataset;

	for (int i = 0; i < size; i++)
	{
		vector<bool> availableFeatures(11, false);
		int pickedFeature;
		for (int j = 0; j < 4; j++)
		{
			pickedFeature = b.getRandomNumberInt(0, 11-1);
			availableFeatures[pickedFeature] = true;
		}

		trees.push_back(RegressionTree(b.getNumberFeatures(), b.getSampleWithReplacementBefore(_learningSampleSize, _endTrainingDataset),
			b.getAttrValues(), b.getIsCategorical(), b.getFeatureNames(), _minSampleSize, _maxDepthTree, _minSD, availableFeatures));
	}
}

void RandomForest::test(Bootstrap b)
{
	vector<pair<double, vector<int>>> testDataset = b.getSmallFixedSample(endValidationDatasetIndex, b.getDatasetSize());
	int S = testDataset.size();
	double testDatasetMean = 0;
	vector<double> metrics = { 0.10, 0.25, 0.50, 0.75 };
	vector<double> randomForestResultsMean(4,0);
	vector<double> randomForestResultsMedian(4,0);

	double forestSquareError = 0;
	double forestSquareErrorMedian = 0;
	for (int i = 0; i < S; i++)
	{
		double predictedPriceSum = 0;
		double currPrediction = 0;
		vector<double> forestResults(size, 0);
		for (int j = 0; j < size; j++)
		{
			currPrediction = trees[j].predictPrice(testDataset[i].second);
			predictedPriceSum += currPrediction;

			double error = currPrediction - testDataset[i].first;
			double errorPercent = error / testDataset[i].first;

			forestResults[j] = currPrediction;
		}
		predictedPriceSum /= size;

		// for mean
		double error = predictedPriceSum - testDataset[i].first;
		double errorPercent = error / testDataset[i].first;

		forestSquareError += (error * error);

		// for median
		double median;
		sort(forestResults.begin(), forestResults.end());
		if (size % 2 == 0)
		{
			median = (forestResults[size / 2] + forestResults[size / 2 - 1]) / 2;
		}
		else
		{
			median = forestResults[size / 2 - 1];
		}

		double errorMedian = median - testDataset[i].first;
		double errorPercentMedian = errorMedian / testDataset[i].first;

		forestSquareErrorMedian += (errorMedian * errorMedian);

		// for mean
		for (int k = 0; k < 4; k++)
		{
			if (errorPercent < metrics[k] && errorPercent > -metrics[k])
			{
				randomForestResultsMean[k]++;
			}
		}

		// check for median
		for (int k = 0; k < 4; k++)
		{
			if (errorPercentMedian < metrics[k] && errorPercentMedian > -metrics[k])
			{
				randomForestResultsMedian[k]++;
			}
		}

		testDatasetMean += testDataset[i].first;
	}

	cout << "forest accuracy - mean:\n";
	for (int i = 0; i < 4; i++)
	{
		// number correct guesses -> ratio correct guesses
		randomForestResultsMean[i] = randomForestResultsMean[i] / S;
		std::cout << setw(12) << randomForestResultsMean[i];
	}
	cout << "\n";
	cout << "forest accuracy - median:\n";
	for (int i = 0; i < 4; i++)
	{
		randomForestResultsMedian[i] = randomForestResultsMedian[i] / S;
		std::cout << setw(12) << randomForestResultsMedian[i];
	}
	cout << "\n";
	cout << "rmse for mean:\n";

	double forestMeanSquareError = forestSquareError / S;
	double forestRootMeanSquareError = sqrt(forestMeanSquareError);
	cout << setw(12) << forestRootMeanSquareError << "\n";

	double forestMeanSquareErrorMedian = forestSquareErrorMedian / S;
	double forestRootMeanSquareErrorMedian = sqrt(forestMeanSquareErrorMedian);
	cout << "rmse for median:\n";
	cout << setw(12) << forestRootMeanSquareErrorMedian << "\n";

	cout << "\n";
	cout << "test dataset mean:\n";
	testDatasetMean = testDatasetMean / S;
	cout << setw(12) << testDatasetMean << "\n\n";
}

void RandomForest::testWithValidationSet(Bootstrap b)
{
	vector<pair<double, vector<int>>> validationDataset = b.getSmallFixedSample(endTrainingDatasetIndex, endValidationDatasetIndex);
	int S = validationDataset.size();

	// calculate trees accuracies
	double testDatasetMean = 0;
	vector<double> metrics = { 0.10, 0.25, 0.50, 0.75 };
	vector<vector<double>> treesResults(size, vector<double>(4, 0));
	double currPrediction;
	for (int i = 0; i < S; i++)
	{
		for (int j = 0; j < size; j++)
		{
			currPrediction = trees[j].predictPrice(validationDataset[i].second);
			double error = currPrediction - validationDataset[i].first;
			double errorPercent = error / validationDataset[i].first;

			for (int k = 0; k < 4; k++)
			{
				if (errorPercent < metrics[k] && errorPercent > -metrics[k])
				{
					treesResults[j][k]++;
				}
			}
		}
	}

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			treesResults[i][j] = treesResults[i][j] / S;
		}
	}

	// calculate weights
	vector<double> selectedWeights(size, 0);
	for (int i = 0; i < size; i++)
	{
		selectedWeights[i] = treesResults[i][1];
	}

	vector<double> weighs = weightResultTrees(selectedWeights);
	vector<double> sqWeights = squareWeightResultTrees(selectedWeights);

	//for (auto i : weighs) cout << i << " "; cout << "\n";
	//for (auto i : sqWeights) cout << i << " "; cout << "\n";

	// calculate accuracy for the forest

	vector<pair<double, vector<int>>> testDataset = b.getSmallFixedSample(endValidationDatasetIndex, b.getDatasetSize());
	int testDatasetSize = testDataset.size();

	vector<double> randomForestResults(4, 0);
	vector<double> randomForestResultsSq(4, 0);
	double forestSquareErrorWeight = 0;
	double forestSquareErrorWeightSq = 0;
	for (int i = 0; i < testDatasetSize; i++)
	{
		double predictedPriceSumWight = 0;
		double predictedPriceSumWightSq = 0;
		double currPrediction = 0;
		for (int j = 0; j < size; j++)
		{
			currPrediction = trees[j].predictPrice(testDataset[i].second);

			predictedPriceSumWight += weighs[j] * currPrediction;
			predictedPriceSumWightSq += sqWeights[j] * currPrediction;
		}

		double errorWeight = predictedPriceSumWight - testDataset[i].first;
		double errorPercentWight = errorWeight / testDataset[i].first;
		forestSquareErrorWeight += (errorWeight * errorWeight);

		double errorWeightSq = predictedPriceSumWightSq - testDataset[i].first;
		double errorPercentWightSq = errorWeightSq / testDataset[i].first;
		forestSquareErrorWeightSq += (errorWeightSq * errorWeightSq);


		// for wight
		for (int k = 0; k < 4; k++)
		{
			if (errorPercentWight < metrics[k] && errorPercentWight > -metrics[k])
			{
				randomForestResults[k]++;
			}
		}

		// for weightSq
		for (int k = 0; k < 4; k++)
		{
			if (errorPercentWightSq < metrics[k] && errorPercentWightSq > -metrics[k])
			{
				randomForestResultsSq[k]++;
			}
		}
	}

	//cout << "accuracy for linear weights:\n";
	//for (int i = 0; i < 4; i++)
	//{
	//	// number correct guesses -> ratio correct guesses
	//	randomForestResults[i] = randomForestResults[i] / testDatasetSize;
	//	std::cout << setw(12) << randomForestResults[i];
	//}
	//cout << "\n";

	cout << "accuracy:\n";
	for (int i = 0; i < 4; i++)
	{
		randomForestResultsSq[i] = randomForestResultsSq[i] / testDatasetSize;
		std::cout << setw(12) << randomForestResultsSq[i];
	}
	cout << "\n";

	//cout << "root mean squared error with linear weights:\n";
	//double forestMeanSquareError = forestSquareErrorWeight / testDatasetSize;
	//double forestRootMeanSquareError = sqrt(forestMeanSquareError);
	//cout << setw(12) << forestRootMeanSquareError << "\n";

	cout << "root mean squared error:\n";
	double forestMeanSquareErrorSq = forestSquareErrorWeightSq / testDatasetSize;
	double forestRootMeanSquareErrorSq = sqrt(forestMeanSquareErrorSq);
	cout << setw(12) << forestRootMeanSquareErrorSq << "\n";
}
