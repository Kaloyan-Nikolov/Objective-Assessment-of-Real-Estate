#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

using namespace std;

class Node
{
private:
	vector<pair<double, vector<int>>> nodeDataset;
	int nodeDatasetSize;
	int attrIndex;
	string attributeName;
	map<int, Node*> children;
	Node* parent;
	double mean;
	double SD;
	bool isCategorical;
	int splitValue;

	void findMean();
	void findSD();

public:
	Node();
	Node(vector<pair<double, vector<int>>>& _nodeDataset, Node* _parent);

	void setNodeDataset(vector<pair<double, vector<int>>>& _nodeDataset);
	void setAttrIndex(int _attrIndex);
	void setAttributeName(string _attributeName);
	void setChildren(map<int, Node*>& _children);
	void setParent(Node* _parent);
	void setMean(double _mean);
	void setSD(double _SD);
	void setIsCategorical(bool _isCategorical);
	void setSplitValue(int _splitValue);

	vector<pair<double, vector<int>>> getNodeDataset();
	int getAttrIndex();
	string getAttributeName();
	map<int, Node*> getChildren();
	Node* getParent();
	double getMean();
	double getSD();
	bool getIsCategorical();
	int getSplitValue();

	void addChild(int key, Node* child);
	void printNode();
};