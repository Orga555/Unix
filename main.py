import numpy as np
from random import randrange
from sklearn.datasets import make_blobs
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def distanceNorm(Norm, D_value):

    if Norm == '1':
        counter = np.absolute(D_value)
        counter = np.sum(counter)
    elif Norm == '2':
        counter = np.power(D_value, 2)
        counter = np.sum(counter)
        counter = np.sqrt(counter)
    elif Norm == 'Infinity':
        counter = np.absolute(D_value)
        counter = np.max(counter)
    else:
        raise Exception('We will program this later......')

    return counter


def fit(features, labels, iter_ratio, k, norm):#k是寻找nearHit和nearMiss的数量 norm是距离公式
    #初始化
    (n_samples, n_features) = np.shape(features)
    distance = np.zeros((n_samples, n_samples))
    weight = np.zeros(n_features)
    labels = list(labels)

    #计算距离
    for index_i in range(n_samples):
        for index_j in range(index_i + 1, n_samples):
            D_value = features[index_i] - features[index_j]
            distance[index_i, index_j] = distanceNorm(norm, D_value)
    distance += distance.T

    #迭代
    for iter_num in range(int(iter_ratio * n_samples)):
        #随机抽取样本
        index_i = randrange(0, n_samples, 1)
        self_features = features[index_i]

        #初始化
        nearHit = list()
        nearMiss = dict()
        n_labels = list(set(labels))
        termination = np.zeros(len(n_labels))
        del n_labels[n_labels.index(labels[index_i])]
        for label in n_labels:
            nearMiss[label] = list()
        distance_sort = list()

        #寻找nearHit和nearMiss
        distance[index_i, index_i] = np.max(distance[index_i])  # filter self-distance
        for index in range(n_samples):
            distance_sort.append([distance[index_i, index], index, labels[index]])

        distance_sort.sort(key=lambda x: x[0])

        for index in range(n_samples):
            #寻找nearHit
            if distance_sort[index][2] == labels[index_i]:
                if len(nearHit) < k:
                    nearHit.append(features[distance_sort[index][1]])
                else:
                    termination[distance_sort[index][2]] = 1
                    #寻找nearMiss
            elif distance_sort[index][2] != labels[index_i]:
                if len(nearMiss[distance_sort[index][2]]) < k:
                    nearMiss[distance_sort[index][2]].append(features[distance_sort[index][1]])
                else:
                    termination[distance_sort[index][2]] = 1

            if list(termination).count(0) == 0:
                break

                #更新权重
        nearHit_term = np.zeros(n_features)
        for x in nearHit:
            nearHit += np.abs(np.power(self_features - x, 2))
        nearMiss_term = np.zeros((len(list(set(labels))), n_features))
        for index, label in enumerate(nearMiss.keys()):
            for x in nearMiss[label]:
                nearMiss_term[index] += np.abs(np.power(self_features - x, 2))
            weight += nearMiss_term[index] / (k * len(nearMiss.keys()))
        weight -= nearHit_term / k

        #返回权重
    return weight / (iter_ratio * n_samples)


def test():
    (features, labels) = make_blobs(n_samples=500, n_features=10, centers=4)#生成数据集
    features = normalize(X=features, norm='l2', axis=0)#数据归一化
    end=0
    for x in range(1, 11):
        weight = fit(features=features, labels=labels, iter_ratio=1, k=5, norm='2')
        print(weight)
        end=weight
    print('最后一次')
    print(end)
    X=[]
    y=[]
    X=np.array(features)
    y=np.array(labels)
    arr=np.array([0,1,2,3,4,5,6,7,8,9])
    X = X[:,arr]
    clf = SVC(kernel='linear')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("SVC")
    print(score)


if __name__ == '__main__':  test();
'''if __name__ == '__main__':
    data = open('./iris.data').readlines()
    X = []
    y = []
    for line in data:
        format_line = line.strip().split(',')
        label = int(format_line[-1])
        X.append(format_line[:-1])
        y.append([label])
    X = np.array(X).astype(float)
    y = np.array(y).astype(int)
    subset = graph_fs(X, y)
    print(subset)'''
