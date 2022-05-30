#include "Bootstrap.h"

bool Bootstrap::readInput()
{
	ifstream myfile("melb-data.csv", ios::in);
	if (!myfile.is_open())
	{
		cerr << "File not found!\n";
		return false;
	}

	string currLine = "";

	int prevPos = 0;
	int pos;
	string currHeading;
	featureNames.resize(NUMBER_FEATURES + 1);
	if (getline(myfile, currLine)) // get header
	{
		prevPos = currLine.find('S');
		for (int i = 0; i < NUMBER_FEATURES + 1; i++)
		{
			pos = currLine.find(',', prevPos);
			if (pos == string::npos) pos = currLine.size();
			currHeading = currLine.substr(prevPos, pos - prevPos);
			featureNames[i] = currHeading;
			prevPos = pos + 1;
		}
	}

	vector<int> numAttrValues(NUMBER_FEATURES, 0);
	attrValues.resize(NUMBER_FEATURES);
	string currFeature;
	while (getline(myfile, currLine))
	{
		
		vector<int> currFeatures(NUMBER_FEATURES);
		int prevPos = 0;
		int pos;
		bool flag = true;
		for (int i = 0; i < NUMBER_FEATURES; i++)
		{
			pos = currLine.find(',', prevPos);
			currFeature = currLine.substr(prevPos, pos - prevPos);

			if (currFeature == "" || (currFeature == "0" && i == 7))
			{
				flag = false;
				break;
				currFeatures[i] = -1;
			}
			else if (isCategorical[i])
			{
				if (attrValues[i].find(currFeature) == attrValues[i].end())
				{
					// add to map
					numAttrValues[i]++;
					attrValues[i][currFeature] = numAttrValues[i];
				}

				currFeatures[i] = attrValues[i][currFeature];
			}
			else
			{
				currFeatures[i] = stoi(currFeature);
			}

			prevPos = pos + 1;
		}

		if (flag) {
			// read price
			pos = currLine.size();
			currFeature = currLine.substr(prevPos, pos - prevPos);
			if (currFeature == "" || currFeature == "0")
			{
				continue;
				cout << "WE CONTINUE\n";
			}
			double currPrice = stoi(currFeature);

			dataset.push_back({ currPrice , currFeatures });
		}
		else
		{
			flag = true;
		}
	}
	datasetSize = dataset.size();

	myfile.close();
	return true;
}

void Bootstrap::initializeGen()
{
	std::random_device rd;
	std::mt19937 _g(rd());
	g = _g;
}

unsigned int Bootstrap::getRandomNumberInt(unsigned int lowerbound, unsigned int upperbound)
{
	std::uniform_int_distribution<unsigned int> distribution(lowerbound, upperbound);
	return distribution(g);
}

Bootstrap::Bootstrap()
{
	readInput();
	initializeGen();
}

vector<pair<double, vector<int>>> Bootstrap::getCompleteDataset()
{
	return dataset;
}

int Bootstrap::getDatasetSize()
{
	return datasetSize;
}

vector<bool> Bootstrap::getIsCategorical()
{
	return isCategorical;
}

vector<map<string, int>> Bootstrap::getAttrValues()
{
	return attrValues;
}

vector<string> Bootstrap::getFeatureNames()
{
	return featureNames;
}

int Bootstrap::getNumberFeatures()
{
	return NUMBER_FEATURES;
}

void Bootstrap::shuffleDataset()
{
	std::random_device rd;
	std::mt19937 _g(rd());
	g = _g;

	std::shuffle(dataset.begin(), dataset.end(), g);
}

vector<pair<double, vector<int>>> Bootstrap::getFixedSampleForTraining(int startTestDataset, int endTestDataset)
{
	vector<pair<double, vector<int>>> sample;
	for (int i = 0; i < datasetSize; i++)
	{
		if (i >= startTestDataset && i <= endTestDataset)
		{
			// we do not learn from the test dataset
			continue;
		}

		sample.push_back(dataset[i]);
	}
	return sample;
}

vector<pair<double, vector<int>>> Bootstrap::getSmallFixedSample(int start, int end)
{
	vector<pair<double, vector<int>>> sample;
	if (end > datasetSize) end = datasetSize;
	for (int i = start; i < end; i++)
	{
		sample.push_back(dataset[i]);
	}
	return sample;
}

vector<pair<double, vector<int>>> Bootstrap::getSampleWithReplacementBefore(int desizedSize, int endTrainingDataset)
{
	vector<pair<double, vector<int>>> sample;
	int nextExampleIndex;
	for (int i = 0; i < desizedSize; i++)
	{
		nextExampleIndex = getRandomNumberInt(0, endTrainingDataset - 1);
		sample.push_back(dataset[nextExampleIndex]);
	}
	return sample;
}
