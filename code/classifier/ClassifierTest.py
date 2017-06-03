from sklearn.model_selection import train_test_split

from classifier.ClassifierClass import Classifier
from datagenerator.DataGenerator import gendata


def main():
    dir_dataset = '../../resource/dataset/fingerspelling5/dataset5/A/'
    n_data = 500
    data, labels = gendata(dir_dataset, n_data)

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.33, random_state=42)

    classifier = Classifier()
    classifier.train(data_train, labels_train)
    labels_predict = classifier.predict(data_test)
    error = 0
    for i, label_predict in enumerate(labels_predict):
        if label_predict != labels_test[i]:
            error += 1
    error /= len(data_test)

    print("Error: " + str(error))


main()