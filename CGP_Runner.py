# -*- coding: utf-8 -*-
# Author: YeXiaona
# Date  : 2020-11-11
# Runner for CGP

import CGP
import data_loader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn import metrics
import csv

'load_data'
train_data_array, test_data_array = data_loader.load_data()
individual_size = 100
uar = []
uf1 = []
acc = []
recall = []
precision = []
f1 = []

data = np.vstack((train_data_array[0], test_data_array[0]))
for individual_index in range(individual_size):
    person_index = 0
    test_tconf = None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # for train_data, test_data in zip(train_data_array, test_data_array):
    for train_index, test_index in skf.split(X=data[:,:-1], y=data[:,-1]):
        train_data, test_data = data[train_index], data[test_index]
        train_X, test_X, train_y, test_y = train_data[:, :-1], test_data[:, :-1], train_data[:, -1], test_data[:, -1]
        scaler = MinMaxScaler()
        scaler.fit(np.vstack((train_X, test_X)))
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

        # cartesion_map = CGP.CartesionMap(4,4,4)
        cartesion_map = CGP.CartesionMap(4,4,2)
        cartesion_map.generateIndividual()

        mapped_train_X_0 = []
        mapped_train_X_1 = []
        mapped_train_X_2 = []
        mapped_train_X_3 = []
        for i in range(len(train_X)):
            temp_mapped_train_X_0 = []
            temp_mapped_train_X_1 = []
            temp_mapped_train_X_2 = []
            temp_mapped_train_X_3 = []
            for j in range(28 * 28 * 3):
                # mapped_results = cartesion_map.get_mapped_results([train_X[i][j+28*28*3],
                #                                                    train_X[i][j+28*28*3*2],
                #                                                    train_X[i][j+28*28*3*3]])
                mapped_results = cartesion_map.get_mapped_results([train_X[i][j + 28 * 28 * 3],
                                                                   train_X[i][j + 28 * 28 * 3 * 2]])
                temp_mapped_train_X_0.append(mapped_results[0])
                temp_mapped_train_X_1.append(mapped_results[1])
                temp_mapped_train_X_2.append(mapped_results[2])
                temp_mapped_train_X_3.append(mapped_results[3])
            mapped_train_X_0.append(temp_mapped_train_X_0)
            mapped_train_X_1.append(temp_mapped_train_X_1)
            mapped_train_X_2.append(temp_mapped_train_X_2)
            mapped_train_X_3.append(temp_mapped_train_X_3)

        mapped_test_X_0 = []
        mapped_test_X_1 = []
        mapped_test_X_2 = []
        mapped_test_X_3 = []
        for i in range(len(test_X)):
            temp_mapped_test_X_0 = []
            temp_mapped_test_X_1 = []
            temp_mapped_test_X_2 = []
            temp_mapped_test_X_3 = []
            for j in range(28 * 28 * 3):
                # mapped_results = cartesion_map.get_mapped_results([test_X[i][j + 28 * 28 * 3],
                #                                                    test_X[i][j + 28 * 28 * 3 * 2],
                #                                                    test_X[i][j + 28 * 28 * 3 * 3]])
                mapped_results = cartesion_map.get_mapped_results([test_X[i][j + 28 * 28 * 3],
                                                                   test_X[i][j + 28 * 28 * 3 * 2]])
                temp_mapped_test_X_0.append(mapped_results[0])
                temp_mapped_test_X_1.append(mapped_results[1])
                temp_mapped_test_X_2.append(mapped_results[2])
                temp_mapped_test_X_3.append(mapped_results[3])
            mapped_test_X_0.append(temp_mapped_test_X_0)
            mapped_test_X_1.append(temp_mapped_test_X_1)
            mapped_test_X_2.append(temp_mapped_test_X_2)
            mapped_test_X_3.append(temp_mapped_test_X_3)

        scaler2 = MinMaxScaler()
        scaler2.fit(np.vstack((mapped_train_X_0, mapped_test_X_0)))
        mapped_train_X_0 = scaler2.transform(mapped_train_X_0)
        mapped_test_X_0 = scaler2.transform(mapped_test_X_0)
        scaler2.fit(np.vstack((mapped_train_X_1, mapped_test_X_1)))
        mapped_train_X_1 = scaler2.transform(mapped_train_X_1)
        mapped_test_X_1 = scaler2.transform(mapped_test_X_1)
        scaler2.fit(np.vstack((mapped_train_X_2, mapped_test_X_2)))
        mapped_train_X_2 = scaler2.transform(mapped_train_X_2)
        mapped_test_X_2 = scaler2.transform(mapped_test_X_2)
        scaler2.fit(np.vstack((mapped_train_X_3, mapped_test_X_3)))
        mapped_train_X_3 = scaler2.transform(mapped_train_X_3)
        mapped_test_X_3 = scaler2.transform(mapped_test_X_3)

        base_estimator = SVC(kernel='rbf', gamma='scale', probability=True, decision_function_shape='ovo')
        predict_y_list = []
        # base_estimator.fit(mapped_train_X_0, train_y)
        # predict_y_list.append(base_estimator.predict(mapped_test_X_0).tolist())
        # base_estimator.fit(mapped_train_X_1, train_y)
        # predict_y_list.append(base_estimator.predict(mapped_test_X_1).tolist())
        # base_estimator.fit(mapped_train_X_2, train_y)
        # predict_y_list.append(base_estimator.predict(mapped_test_X_2).tolist())
        # base_estimator.fit(mapped_train_X_3, train_y)
        # predict_y_list.append(base_estimator.predict(mapped_test_X_3).tolist())
        GA = CGP.GA
        fs = GA.generateFS(np.hstack((train_X[:,:28*28*3], mapped_train_X_0)), train_y)
        base_estimator.fit(np.hstack((train_X[:,:28*28*3], mapped_train_X_0))[fs], train_y)
        predict_y_list.append(base_estimator.predict(np.hstack((test_X[:,:28*28*3], mapped_test_X_0))[fs]).tolist())
        fs = GA.generateFS(np.hstack((train_X[:, :28 * 28 * 3], mapped_train_X_1)), train_y)
        base_estimator.fit(np.hstack((train_X[:,:28*28*3], mapped_train_X_1))[fs], train_y)
        predict_y_list.append(base_estimator.predict(np.hstack((test_X[:,:28*28*3], mapped_test_X_1))[fs]).tolist())
        fs = GA.generateFS(np.hstack((train_X[:, :28 * 28 * 3], mapped_train_X_2)), train_y)
        base_estimator.fit(np.hstack((train_X[:,:28*28*3], mapped_train_X_2))[fs], train_y)
        predict_y_list.append(base_estimator.predict(np.hstack((test_X[:,:28*28*3], mapped_test_X_2))[fs]).tolist())
        fs = GA.generateFS(np.hstack((train_X[:, :28 * 28 * 3], mapped_train_X_3)), train_y)
        base_estimator.fit(np.hstack((train_X[:,:28*28*3], mapped_train_X_3))[fs], train_y)
        predict_y_list.append(base_estimator.predict(np.hstack((test_X[:,:28*28*3], mapped_test_X_3))[fs]).tolist())

        predict_y_list = np.array(predict_y_list)

        voted_y = []
        tieNumber = 0
        for i in range(predict_y_list.shape[1]):
            ylist = predict_y_list[:,i]
            dict = {0:0, 1:0, 2:0}
            for y in ylist:
                dict[y] += 1
            maxvote = max(dict, key=dict.get)
            if dict[0] == dict[1] | dict[0] == dict[2] | dict[1] == dict[2]:
                tieNumber += 1
            voted_y.append(maxvote)
        print("person %s: 平票概率为%s"%(person_index, tieNumber/len(test_y)))

        f1score = metrics.f1_score(test_y, voted_y, average='macro')
        # accuracy = metrics.f1_score(test_y, voted_y)
        conf = metrics.confusion_matrix(test_y, voted_y, labels=[0, 1, 2])
        print("f1score: %s  " % (f1score))
        print(conf)

        if person_index == 0:
            test_tconf = np.array(conf)
        else:
            test_tconf += np.array(conf)

        person_index += 1

    print('---------------------------------------------------')
    print("the %s times individual result:" % (individual_index))
    UAR = np.mean([test_tconf[i][i] / sum(test_tconf[i]) for i in range(3)])
    UF1 = sum([2 * test_tconf[i][i] / (sum(test_tconf[:, i]) + sum(test_tconf[i])) for i in range(3)]) / 3.0

    Acc = sum([test_tconf[i][i] for i in range(3)]) / np.sum(test_tconf)
    Recall = np.mean([test_tconf[i][i] / sum(test_tconf[i]) if sum(test_tconf[i]) != 0 else 0 for i in range(3)])
    Precision = np.mean(
        [test_tconf[i][i] / sum(test_tconf[:, i]) if sum(test_tconf[:, i]) != 0 else 0 for i in range(3)])
    F1 = 2 * Recall * Precision / sum([Recall, Precision])

    print('test total_confusion\n {}'.format(test_tconf))
    print('test UAR {}, UF1 {}, Acc {}'.format(round(UAR, 4), round(UF1, 4), round(Acc, 4)))
    print('test Recall {}, Precision {}, F1 {}'.format(round(Recall, 4), round(Precision, 4), round(F1, 4)))

    uar.append(UAR)
    uf1.append(UF1)
    acc.append(Acc)
    f1.append(F1)
    recall.append(Recall)
    precision.append(Precision)

f = open("./individualResults.csv", "w")
csv_writer = csv.writer(f, lineterminator='\n')
csv_writer.writerow(np.hstack(("UAR", uar)))
csv_writer.writerow(np.hstack(("UF1", uf1)))
csv_writer.writerow(np.hstack(("ACC", acc)))
csv_writer.writerow(np.hstack(("RECALL", recall)))
csv_writer.writerow(np.hstack(("PRECISION", precision)))
csv_writer.writerow(np.hstack(("F1SCORE", f1)))
f.close()
print("================================")
print('test UAR {}, UF1 {}, Acc {}'.format(round(np.mean(uar), 4), round(np.mean(uf1), 4), round(np.mean(acc), 4)))
print('test Recall {}, Precision {}, F1 {}'.format(round(np.mean(recall), 4), round(np.mean(precision), 4), round(np.mean(f1), 4)))