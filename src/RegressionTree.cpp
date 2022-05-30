#include "RegressionTree.h"

void RegressionTree::initializeGen()
{
	std::random_device rd;
	std::mt19937 _g(rd());
	g = _g;
}

unsigned int RegressionTree::getRandomNumberInt(unsigned int lowerbound, unsigned int upperbound)
{
	std::uniform_int_distribution<unsigned int> distribution(lowerbound, upperbound);
	return distribution(g);
}

// {mean, number examples}
pair<double,int> RegressionTree::findNumberExamplesAndSDcorrespondingData(int searchedValue, 
	int attrNum, vector<pair<double, vector<int>>>& trainingData)
{
	int S = trainingData.size();
	pair<double, int> result;
	for (int i = 0; i < S; i++)
	{
		if (trainingData[i].second[attrNum] == searchedValue)
		{
			result.first += trainingData[i].first;
			result.second++;
		}
	}
	result.first /= result.second;
	return result;
}

double RegressionTree::findSD(vector<pair<double, vector<int>>>& trainingData, double mean, int searchedValue, int attrNum)
{
	double SSR = 0; // sum of squared residuals
	double S = trainingData.size();
	for (int i = 0; i < S; i++)
	{
		if (trainingData[i].second[attrNum] == searchedValue)
		{
			SSR += pow(trainingData[i].first - mean, 2);
		}
	}
	double SD = sqrt(SSR / S);
	return SD;
}

double RegressionTree::findWeightedSD(const int attrNum, vector<pair<double, vector<int>>>& trainingData)
{
	double weightedSD = 0;
	int S = trainingData.size();
	for (auto it = attrValues[attrNum].begin(); it != attrValues[attrNum].end(); it++)
	{
		int searchedValue = it->second;

		pair<double, int> correspData = findNumberExamplesAndSDcorrespondingData(searchedValue, attrNum, trainingData);
		double mean = correspData.first;
		int numberExamples = correspData.second;

		double P = (double)numberExamples / S;
		double currSD = findSD(trainingData, mean, searchedValue, attrNum);
		
		if (P > 0)
		{
			weightedSD += P * currSD;
		}
	}

	return weightedSD;
}

vector<pair<int, double>> RegressionTree::makeDataConcise(vector<pair<double, vector<int>>>& trainingData, int attrNum)
{
	int S = trainingData.size();
	vector<pair<int, double>> conciseData;
	for (int i = 0; i < S; i++)
	{
		conciseData.push_back({ trainingData[i].second[attrNum], trainingData[i].first });
	}

	return conciseData;
}

pair<double, int> RegressionTree::findBestSplit(vector<pair<int, double>>& conciseData)
{
	double minWeightedSD = INT_MAX;
	int bestSplitValue;
	double S = conciseData.size();
	for (int i = 0; i < S - 1; i++)
	{
		if (i > 1 && conciseData[i].first == conciseData[i - 1].first)
		{
			continue;
		}

		double currWeightedSD = evaluateSplit(conciseData, conciseData[i].first);
		if (minWeightedSD > currWeightedSD)
		{
			minWeightedSD = currWeightedSD;
			bestSplitValue = conciseData[i].first;
		}
	}
	return { minWeightedSD,  bestSplitValue };
}

// we evaluate by finding the 2 means, 2 sd-s and then the weighted SD.
double RegressionTree::evaluateSplit(vector<pair<int, double>>& conciseData, int splitValue)
{
	pair<double, double> sums;
	pair<int, int> numberExamples;
	int S = conciseData.size();
	int cnt = 0;
	while (cnt < S && conciseData[cnt].first <= splitValue)
	{
		numberExamples.first++;
		sums.first += conciseData[cnt].second;
		cnt++;
	}
	while (cnt < S)
	{
		numberExamples.second++;
		sums.second += conciseData[cnt].second;
		cnt++;
	}
	pair<double, double> means = { sums.first / numberExamples.first, sums.second / numberExamples.second };

	cnt = 0;
	pair<double, double> SSRs = { 0,0 };
	while (cnt < S && conciseData[cnt].first <= splitValue)
	{
		SSRs.first += pow(conciseData[cnt].second - means.first, 2);
		cnt++;
	}
	while (cnt < S)
	{
		SSRs.second += pow(conciseData[cnt].second - means.second, 2);
		cnt++;
	}

	pair<double, double> SDs = { sqrt(SSRs.first / numberExamples.first), sqrt(SSRs.second / numberExamples.second) };

	pair<double, double> Ps = { (double)numberExamples.first / S, (double)numberExamples.second / S };
	double weightedSD = 0;
	if (Ps.first > 0 && SDs.first > 0)
	{
		weightedSD += Ps.first * SDs.first;
	}
	if (Ps.second > 0 && SDs.second > 0)
	{
		weightedSD += Ps.second * SDs.second;
	}

	return weightedSD;
}

// return {reduction SD, split value to achieve it}
pair<double, int> RegressionTree::findReductionSD(vector<pair<double, vector<int>>>& trainingData, Node * node, const int attrNum)
{
	if (isCategorical[attrNum])
	{
		double weightedSD = findWeightedSD(attrNum, trainingData);
		return { node->getSD() - weightedSD, -1 };
	}

	// price, attr
	vector<pair<int, double>> conciseData = makeDataConcise(trainingData, attrNum);
	sort(conciseData.begin(), conciseData.end());

	pair<double, int> bestSplit = findBestSplit(conciseData);
	double weightedSD = bestSplit.first;
	int splitValue = bestSplit.second;

	return { node->getSD() - weightedSD, splitValue };
}

int RegressionTree::findAttrMaxReductionSD(vector<pair<double, vector<int>>>& trainingData, Node* node, set<int>& usedAttr)
{
	// choose new available features for every split
	availableFeatures = vector<bool>(11, false);
	int pickedFeature;
	for (int j = 0; j < 4; j++)
	{
		pickedFeature = getRandomNumberInt(0, 11 - 1);
		availableFeatures[pickedFeature] = true;
	}

	double maxReductionSD = -1;
	int indexMaxReductionSD = -1;
	for (int i = 0; i < numberFeatures; i++)
	{
		if (usedAttr.count(i) == 1 || !availableFeatures[i]) // it is already used
		{
			continue;
		}

		pair<double, int> currReductionRes = findReductionSD(trainingData, node, i);
		double currReduction = currReductionRes.first;
		int currSplitValue = currReductionRes.second;
		if (maxReductionSD < currReduction && currReduction >= minSD)
		{
			maxReductionSD = currReduction;
			indexMaxReductionSD = i;
			node->setSplitValue(currSplitValue);
		}
	}
	return indexMaxReductionSD;
}

RegressionTree::RegressionTree()
{
	numberFeatures = 0;
	initializeGen();
}

RegressionTree::RegressionTree(int _numberFeatures, vector<pair<double, vector<int>>> _dataset, 
	vector<map<string, int>> _attrValues, vector<bool> _isCategorical, vector<string> _featureNames,
	int _minSampleSize, int _maxDepthTree, int _minSD, vector<bool> _availableFeatures)
{
	numberFeatures = _numberFeatures;
	attrValues = _attrValues;
	isCategorical = _isCategorical;
	featureNames = _featureNames;
	minSampleSize = _minSampleSize;
	maxDepthTree = _maxDepthTree;
	minSD = _minSD;
	availableFeatures = _availableFeatures;
	initializeGen();

	set<int> usedAttrs;
	root = trainModel(_dataset, usedAttrs);
}

// filters the data based on the value 'searchedValue' of attribute 'attrNum'
vector<pair<double, vector<int>>> RegressionTree::filterData(vector<pair<double, vector<int>>>& trainingData, int attrNum, int searchedValue)
{
	vector<pair<double, vector<int>>> filteredDataset;
	int sz = trainingData.size();
	for (int i = 0; i < sz; i++)
	{
		if (trainingData[i].second[attrNum] == searchedValue)
			filteredDataset.push_back(trainingData[i]);
	}
	return filteredDataset;
}

// filters the data based on the value 'splitValue' of the node for attribute 'attrNum' // smaller
vector<pair<double, vector<int>>> RegressionTree::filterDataS(vector<pair<double, vector<int>>>& trainingData, int attrNum, int splitValue)
{
	vector<pair<double, vector<int>>> filteredDataset;
	int sz = trainingData.size();
	for (int i = 0; i < sz; i++)
	{
		if (trainingData[i].second[attrNum] <= splitValue)
			filteredDataset.push_back(trainingData[i]);
	}
	return filteredDataset;
}

// filters the data based on the value 'splitValue' of the node for attribute 'attrNum' // larger
vector<pair<double, vector<int>>> RegressionTree::filterDataL(vector<pair<double, vector<int>>>& trainingData, int attrNum, int splitValue)
{
	vector<pair<double, vector<int>>> filteredDataset;
	int sz = trainingData.size();
	for (int i = 0; i < sz; i++)
	{
		if (trainingData[i].second[attrNum] > splitValue)
			filteredDataset.push_back(trainingData[i]);
	}
	return filteredDataset;
}

Node * RegressionTree::trainModel(vector<pair<double, vector<int>>>& trainingData, set<int> usedAttrs, int currDepth, Node * currParent)
{
	Node* node = new Node(trainingData, currParent);

	// pure sample          // all attr are used                      // pre-pruning                           // pre-pruning
	if (node->getSD() < 0.001 || trainingData.size() == numberFeatures || trainingData.size() < minSampleSize || currDepth >= maxDepthTree)
	{
		node->setAttributeName("leaf");
		return node;
	}

	int indexAttrMaxReductionSD = findAttrMaxReductionSD(trainingData, node, usedAttrs);
	// pre-pruning
	if (indexAttrMaxReductionSD == -1)
	{
		node->setAttributeName("leaf");
		return node;
	}

	node->setAttrIndex(indexAttrMaxReductionSD);
	node->setAttributeName(featureNames[indexAttrMaxReductionSD]);

	// creates a child for every value of attribute 'indexAttrMaxGain'
	if (isCategorical[indexAttrMaxReductionSD])
	{
		usedAttrs.insert(indexAttrMaxReductionSD);
		for (auto it = attrValues[indexAttrMaxReductionSD].begin(); it != attrValues[indexAttrMaxReductionSD].end(); it++)
		{
			auto filteredData = filterData(trainingData, indexAttrMaxReductionSD, it->second);
			if (filteredData.size() > 0)
			{
				node->addChild(it->second, trainModel(filteredData, usedAttrs, currDepth + 1));
			}
		}
	}
	else
	{
		//usedAttrs.insert(indexAttrMaxReductionSD);
		auto filteredDataS = filterDataS(trainingData, indexAttrMaxReductionSD, node->getSplitValue());
		if (filteredDataS.size() > 0)
		{
			node->addChild(1, trainModel(filteredDataS, usedAttrs, currDepth + 1));
		}
		auto filteredDataL = filterDataL(trainingData, indexAttrMaxReductionSD, node->getSplitValue());
		if (filteredDataL.size() > 0)
		{
			node->addChild(2, trainModel(filteredDataL, usedAttrs, currDepth + 1));
		}
	}

	return node;
}

void RegressionTree::printRegressionTreeBFS()
{
	if (root == nullptr)
		return;

	queue<Node*> nodes;
	nodes.push(root);
	Node* temp;
	int nodesInLevel = 1;
	int nodesPrintedOnLevel = 0;
	int currLevel = 0;
	cout << "REGRESSION TREE\n";
	cout << "CURRENT LEVEL: " << currLevel << "\n";
	while (!nodes.empty())
	{
		temp = nodes.front();
		nodes.pop();
		nodesPrintedOnLevel++;

		cout << "CURRENT LEVEL: " << currLevel << "\n";
		temp->printNode();

		for (const auto& child : temp->getChildren())
		{
			nodes.push(child.second);
		}

		if (nodesPrintedOnLevel == nodesInLevel)
		{
			nodesInLevel = nodes.size();
			nodesPrintedOnLevel = 0;
			currLevel++;
			if (nodesInLevel > 0)
			{
				cout << "CURRENT LEVEL: " << currLevel << "\n";
			}
		}
	}
}

double RegressionTree::predictPrice(vector<int> example)
{
	Node* copy = root;
	string currAttrName = copy->getAttributeName();
	while (currAttrName != "leaf")
	{
		int attribute = example[copy->getAttrIndex()];
		if (copy->getChildren().count(attribute) == 1)
		{
			copy = copy->getChildren()[attribute];
			currAttrName = copy->getAttributeName();
		}
		else
		{
			break;
		}
	}

	return copy->getMean();
}

double RegressionTree::getStartingSD()
{
	return root->getSD();
}
