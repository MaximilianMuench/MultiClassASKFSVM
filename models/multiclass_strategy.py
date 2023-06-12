from collections import Counter
import numpy as np

class OneVsOneClassifier:
    def __init__(self, BinaryModelClass, **kwargs):
        self.BinaryModelClass = BinaryModelClass
        self.kwargs = kwargs
        self.models = {}

    def fit(self, Ks, y):
        self.classes = np.unique(y)
        for i in range(len(self.classes)):
            for j in range(i + 1, len(self.classes)):
                indices = np.where((y == self.classes[i]) | (y == self.classes[j]))[0]
                Ks_binary=[]
                for K in Ks:
                    K_binary = K[np.ix_(indices, indices)]
                    Ks_binary.append(K_binary)
                y_binary = y[indices]
                model = self.BinaryModelClass(**self.kwargs)
                model.fit(Ks_binary, y_binary)
                self.models[(self.classes[i], self.classes[j])] = model

    def predict(self, Ks_test):
        predictions = []
        for idx in range(Ks_test[0].shape[0]):
            votes = []
            for (class_i, class_j), model in self.models.items():
                K_test_single = [K_test[idx, :] for K_test in Ks_test]
                pred = model.predict(K_test_single)
                votes.append(pred[0])
            most_common = Counter(votes).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)


#
# class OneVsRestClassifier:
#     def __init__(self, BinaryModelClass, **kwargs):
#         self.BinaryModelClass = BinaryModelClass
#         self.models = {}
#         self.kwargs = kwargs
#
#     def fit(self, K, y):
#         self.classes = np.unique(y)
#         for i in self.classes:
#             y_binary = np.where(y == i, 1, -1)
#             model = self.BinaryModelClass()  # create a new instance for each class
#             model.fit(K, y_binary)
#             self.models[i] = model
#
#     def predict(self, K):
#         predictions = []
#         for idx in range(K.shape[0]):
#             scores = []
#             for class_i, model in self.models.items():
#                 score = model.decision_function(K[idx, :])
#                 scores.append((score, class_i))
#             # Remap 1 and -1 to original class labels
#             pred_label = self.class_map[np.sign(max(scores)[0])]
#             predictions.append(pred_label)
#         return np.array(predictions)

class OneVsRestClassifier:
    def __init__(self, BinaryModelClass, **kwargs):
        self.BinaryModelClass = BinaryModelClass
        self.models = {}
        self.class_map = {}
        self.kwargs = kwargs


    def fit(self, K, y):
        self.classes = np.unique(y)
        for idx, i in enumerate(self.classes):
            y_binary = np.where(y == i, 1, -1)
            model = self.BinaryModelClass()  # create a new instance for each class
            model.fit(K, y_binary)
            self.models[i] = model
            self.class_map[idx + 1] = i  # 1 maps to first class, -1 maps to rest

    def predict(self, K):
        predictions = []
        score_matrix = np.zeros(shape=(len(K[0]),len(self.models)))
        for class_i, model in self.models.items():
            scores = model.decision_function(K)
            score_matrix[:, class_i] = scores
        pause = "pause"
        predictions = np.argmax(score_matrix, axis=1)

        return predictions