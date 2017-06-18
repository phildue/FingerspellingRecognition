from sklearn.model_selection import train_test_split, cross_val_score


def hold_out_eval(classifier, data, labels):
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.33, random_state=42)

    classifier.train(data_train, labels_train)
    labels_predict = classifier.predict(data_test)
    error = 0
    for i, label_predict in enumerate(labels_predict):
        if label_predict != labels_test[i]:
            error += 1
    return error / len(data_test)


def crossval(classifier, data, labels,n_jobs=4):
    return cross_val_score(classifier.scikit_object, data, labels.ravel(), n_jobs=n_jobs)
