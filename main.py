import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def getData():
    # Preprocess data to make run time faster
    if not os.path.exists('data.txt'):
        instance_class = np.loadtxt('SpamInstances.txt', delimiter=' ', skiprows=1, usecols=(0, 1))
        feature_vectors = np.genfromtxt('SpamInstances.txt', dtype='str', skip_header=1, usecols=(2,))
        data = np.array([])

        for i in range(feature_vectors.size):
            data_row = np.array([])
            for ch in feature_vectors[i]:
                data_row = np.append(data_row, int(ch))
            data = np.append(data, data_row)
        data = data.reshape(15498, 334)
        data = np.append(instance_class, data, axis=1)
        with open('data.txt', 'a+') as outfile:
            np.savetxt(outfile, data, delimiter=',')
    else:
        data = np.loadtxt('data.txt', delimiter=',')
        return data


def get_probability(X, Y):
    # First row: Number of times a word occurs when there is spam
    # Second row: Number of times a word doesn't occur when there is spam
    yes_spam = np.zeros(2 * 334).reshape(2, 334)
    spam_count = 0
    no_spam = np.zeros(2 * 334).reshape(2, 334)
    no_spam_count = 0

    for i in range(Y.size):
        if Y[i] == 1:
            spam_count = spam_count + 1
            for j in range(X[i, :].size):
                if X[i, j] == 1:
                    yes_spam[0, j] = yes_spam[0, j] + 1
                else:
                    yes_spam[1, j] = yes_spam[1, j] + 1
        else:
            no_spam_count = no_spam_count + 1
            for j in range(X[i, :].size):
                if X[i, j] == 1:
                    no_spam[0, j] = no_spam[0, j] + 1
                else:
                    no_spam[1, j] = no_spam[1, j] + 1

    yes_spam = yes_spam / spam_count
    no_spam = no_spam / no_spam_count
    spam_prob = spam_count / Y.size
    no_spam_prob = no_spam_count / Y.size
    return yes_spam, no_spam, spam_prob, no_spam_prob


def classify(yes_spam, no_spam, spam_prob, no_spam_prob, X):
    # Now get the corresponding probabilities
    # Start with spam probability then check no spam probability
    # Ex: when classifying yes, if a given feature is one, get the number of times the feature is 1 when there is spam
    # else get the number of times it is zero

    # Get conditional independent probabilities
    yes_probabilities = np.array([])
    no_probabilities = np.array([])
    for i in range(X[:, 0].size):
        yes_probability_vec = np.array([])
        no_probability_vec = np.array([])
        for j in range(X[i, :].size):
            if X[i, j] == 1:
                yes_probability_vec = np.append(yes_probability_vec, yes_spam[0, j])
                no_probability_vec = np.append(no_probability_vec, no_spam[0, j])
            else:
                yes_probability_vec = np.append(yes_probability_vec, yes_spam[1, j])
                no_probability_vec = np.append(no_probability_vec, no_spam[1, j])
        yes_probabilities = np.append(yes_probabilities, spam_prob * np.prod(yes_probability_vec))
        no_probabilities = np.append(no_probabilities, no_spam_prob * np.prod(no_probability_vec))

    return yes_probabilities, no_probabilities


def validate(yes_spam, no_spam, spam_prob, no_spam_prob, X, Y):
    # Get conditional probabilities
    yes_prob, no_prob = classify(yes_spam, no_spam, spam_prob, no_spam_prob, X)

    # Have the highest probability be the classification
    classification = np.array([])
    for i in range(yes_prob.size):
        if yes_prob[i] > no_prob[i]:
            classification = np.append(classification, 1)
        else:
            classification = np.append(classification, -1)

    # Get statistics
    tp_rate = 0
    fp_rate = 0
    tn_rate = 0
    fn_rate = 0
    for i in range(Y.size):
        if Y[i] == 1 and classification[i] == 1:
            tp_rate = tp_rate + 1
        elif Y[i] == 1 and classification[i] != 1:
            fn_rate = fn_rate + 1
        elif Y[i] == -1 and classification[i] == -1:
            tn_rate = tn_rate + 1
        elif Y[i] == -1 and classification[i] != -1:
            fp_rate = fp_rate + 1
    accuracy = (tp_rate + tn_rate) / Y.size
    precision = tp_rate / (fp_rate + tp_rate)
    recall = tp_rate / (fn_rate + tp_rate)
    specificity = fp_rate / (fp_rate + tp_rate)
    sensitivity = precision

    return accuracy, precision, recall, specificity, sensitivity


def graph_results(accuracy, precision, recall, specificity, sensitivity, instances):
    # Graph ROC Curve
    plt.xlabel("1 - specificity")
    plt.ylabel('sensitivity')
    plt.title('ROC Curve')
    plt.scatter(recall, precision)
    plt.show()

    # Graph Accuracy over time
    plt.xlabel("Instances")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Graph")
    plt.plot(np.arange(instances), accuracy)
    plt.show()

    # Graph Precision over time
    plt.xlabel("Instances")
    plt.ylabel("Precision")
    plt.title("Precision Graph")
    plt.plot(np.arange(instances), precision)
    plt.show()

    # Graph Recall over time
    plt.xlabel("Instances")
    plt.ylabel("Recall")
    plt.title("Recall Graph")
    plt.plot(np.arange(instances), recall)
    plt.show()


def main():
    # Data ordered as: Observation | Spam | 334 word Yes/No Occurances
    data = getData()
    X = data[:, 2:]
    Y = data[:, 1]

    accuracy = np.array([])
    precision = np.array([])
    recall = np.array([])
    specificity = np.array([])
    sensitivity = np.array([])

    size = 100
    instances = 21
    for i in range(instances):
        # Start with 100 in training set. Increase by 100 20 times. After 20 instances do standard 80/20 split
        if i < 20:
            train_size = size / data[:, 0].size
        else:
            train_size = 0.8

        # Get conditional independence and validate testing data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size)
        yes_spam, no_spam, spam_prob, no_spam_prob = get_probability(X_train, Y_train)
        acc_val, prec_val, rec_val, spec_val, sens_val = validate(yes_spam, no_spam, spam_prob, no_spam_prob, X_test, Y_test)

        # Add new statistics to respective arrays
        accuracy = np.append(accuracy, acc_val)
        precision = np.append(precision, prec_val)
        recall = np.append(recall, rec_val)
        specificity = np.append(specificity, spec_val)
        sensitivity = np.append(sensitivity, sens_val)
        print("Training Instance", i)
        print("Accuracy", acc_val)
        print("Precision", prec_val)
        print("Recall", rec_val, "\n\n")

        # Increase training size by 100 for next iteration
        size = size + 100

    # Graph the results
    graph_results(accuracy, precision, recall, specificity, sensitivity, instances)


if __name__ == "__main__":
    main()

