#include "Node.h"

void Node::findMean()
{
	double sum = 0;
	for (int i = 0; i < nodeDatasetSize; i++)
	{
		sum += nodeDataset[i].first;
	}
	mean = sum / nodeDatasetSize;
}

void Node::findSD()
{
	double SSR = 0; // sum of squared residuals
	for (int i = 0; i < nodeDatasetSize; i++)
	{
		SSR += pow(nodeDataset[i].first - mean, 2);
	}
	SD = sqrt(SSR / nodeDatasetSize);
}

Node::Node()
{
}

Node::Node(vector<pair<double, vector<int>>>& _nodeDataset, Node * _parent)
{
	nodeDataset = _nodeDataset;
	nodeDatasetSize = nodeDataset.size();
	parent = _parent;
	findMean();
	findSD();
}

void Node::setNodeDataset(vector<pair<double, vector<int>>>& _nodeDataset)
{
	nodeDataset = _nodeDataset;
}

void Node::setAttrIndex(int _attrIndex)
{
	attrIndex = _attrIndex;
}

void Node::setAttributeName(string _attributeName)
{
	attributeName = _attributeName;
}

void Node::setChildren(map<int, Node*>& _children)
{
	children = _children;
}

void Node::setParent(Node * _parent)
{
	parent = _parent;
}

void Node::setMean(double _mean)
{
	mean = _mean;
}

void Node::setSD(double _SD)
{
	SD = _SD;
}

void Node::setIsCategorical(bool _isCategorical)
{
	isCategorical = _isCategorical;
}

void Node::setSplitValue(int _splitValue)
{
	splitValue = _splitValue;
}

vector<pair<double, vector<int>>> Node::getNodeDataset()
{
	return nodeDataset;
}

int Node::getAttrIndex()
{
	return attrIndex;
}

string Node::getAttributeName()
{
	return attributeName;
}

map<int, Node*> Node::getChildren()
{
	return children;
}

Node * Node::getParent()
{
	return parent;
}

double Node::getMean()
{
	return mean;
}

double Node::getSD()
{
	return SD;
}

bool Node::getIsCategorical()
{
	return isCategorical;
}

int Node::getSplitValue()
{
	return splitValue;
}

void Node::addChild(int key, Node * child)
{
	children[key] = child;
}

void Node::printNode()
{
	cout << "nodeDatasetSize: " << nodeDatasetSize << "\n";
	cout << "attrIndex: " << attrIndex << "\n";
	cout << "attrName: " << attributeName << "\n";
	cout << "mean: " << mean << "\n";
	cout << "SD: " << SD << "\n";
	cout << "splitValue: " << splitValue << "\n";
	cout << "Childer values: ";
	for (auto it = this->children.begin(); it != this->children.end(); it++)
	{
		cout << it->first << " ";
	}
	cout << "\n\n";
}
