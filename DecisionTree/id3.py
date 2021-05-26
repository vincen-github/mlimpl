import numpy as np
import operator

def createDataSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    dataSet = np.array([[0, 0, 0, 0, 'N'],
                       [0, 0, 0, 1, 'N'],
                       [1, 0, 0, 0, 'Y'],
                       [2, 1, 0, 0, 'Y'],
                       [2, 2, 1, 0, 'Y'],
                       [2, 2, 1, 1, 'N'],
                       [1, 2, 1, 1, 'Y']])
    labels = np.array(['outlook', 'temperature', 'humidity', 'windy'])
    return dataSet, labels

def createTestSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    testSet = np.array([[0, 1, 0, 0],
               [0, 2, 1, 0],
               [2, 1, 1, 0],
               [0, 1, 1, 1],
               [1, 1, 0, 1],
               [1, 0, 1, 0],
               [2, 1, 0, 1]])
    return testSet

def dataset_entropy(dataset):
    classLabel = dataset[:,-1]
    labelCount = {}
    for i in range(classLabel.size):
        label = classLabel[i]
        labelCount[label]=labelCount.get(label,0)+1
    #熵值
    ent=0
    for k,v in labelCount.items():
        ent+=-v/classLabel.size*np.log2(v/classLabel.size)
    return ent

def splitDataSet(dataset,featureIndex,value):
    subdataset = []
    for example in dataset:
        if example[featureIndex]==value:
            subdataset.append(example)
    return np.delete(subdataset,featureIndex,axis=1)

def chooseBestFeature(dataset,labels):
    #特征的个数
    featureNum = labels.size
    #最小熵值
    minEntropy,bestFeatureIndex=1,None
    #样本的总数
    n = dataset.shape[0]
    for i in range(featureNum):
        #指定特征的条件熵
        featureEntropy =0
        #返回所有子集
        featureList = dataset[:, i]
        featureValues = set(featureList)
        for value in featureValues:
            subDataSet = splitDataSet(dataset,i,value)
            featureEntropy+=subDataSet.shape[0]/n*dataset_entropy(subDataSet)
        if minEntropy>featureEntropy:
            minEntropy = featureEntropy
            bestFeatureIndex = i
    print(minEntropy)
    return bestFeatureIndex

def mayorClass(classList):
    labelCount = {}
    for i in range(classList.size):
        label = classList[i]
        labelCount[label] = labelCount.get(label, 0) + 1
    sortedLabel = sorted(labelCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedLabel[0][0]

def createTree(dataset,labels):
    classList = dataset[:,-1]
    if len(set(classList))==1:
        return dataset[:,-1][0]
    if labels.size==0:
        return mayorClass(classList)
    bestFeatureIndex = chooseBestFeature(dataset,labels)
    bestFeature = labels[bestFeatureIndex]
    dtree = {bestFeature:{}}
    featureList = dataset[:,bestFeatureIndex]
    featureValues = set(featureList)
    for value in featureValues:
        subdataset = splitDataSet(dataset,bestFeatureIndex,value)
        sublabels = np.delete(labels,bestFeatureIndex)
        dtree[bestFeature][value]=createTree(subdataset,sublabels)
    return dtree

def predict(tree,labels,testData):
    rootName = list(tree.keys())[0]
    rootValue = tree[rootName]
    featureIndex = list(labels).index(rootName)
    classLabel = None
    for key in rootValue.keys():
        if testData[featureIndex] == int(key):
            if type(rootValue[key]).__name__=="dict":
                classLabel = predict(rootValue[key],labels,testData)
            else:
                classLabel = rootValue[key]
    return classLabel

def predictAll(tree,labels,testSet):
    classLabels = []
    for i in testSet:
        classLabels.append(predict(tree,labels,i))
    return classLabels

if __name__=="__main__":
    dataset,labels = createDataSet()
    # print(dataset_entropy(dataset))
    # s = splitDataSet(dataset,0)
    # for item in s:
    #     print(item)
    tree = createTree(dataset,labels)
    testSet = createTestSet()
    print(predictAll(tree,labels,testSet))
    import decision_tree.treePlotter as treePlotter
    treePlotter.createPlot(tree)