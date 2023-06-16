import gc
import os
import numpy as np
import ray
from collections import Counter
from joblib import Parallel, delayed, parallel_backend
from models.askfsvm_binary import ASKFSVMBinary
from multiprocessing import shared_memory
from timeit import default_timer as timer
from sys import platform

if platform == "linux" or platform == "linux2":
    os.sched_getaffinity(0)

# MP_LIB = "ray"
MP_LIB = "joblib"

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

# inline definition for now

@ray.remote
def fit_remote(K, y, idx, i, max_iter, subsample_size):
    y_binary = np.where(y == i, 1, -1)
    model = ASKFSVMBinary(max_iter=max_iter, subsample_size=subsample_size)
    model.fit(K, y_binary)
    return i, idx, model


def fit_shared(y, idx, i, max_iter, subsample_size, sm_shape, sm_dtype):
    y_binary = np.where(y == i, 1, -1)
    # access shared memory instance
    existing_shm = shared_memory.SharedMemory(name='shared_ndarray')
    data_sm_ = np.ndarray(sm_shape, dtype=sm_dtype, buffer=existing_shm.buf)
    # initialize model
    model = ASKFSVMBinary(max_iter=max_iter, subsample_size=subsample_size)
    with parallel_backend("loky", n_jobs=-1):
        model.fit(data_sm_, y_binary)
    existing_shm.close()
    return i, idx, model


class OneVsRestClassifier:
    def __init__(self, **kwargs):
        self.models = {}
        self.class_map = {}
        self.kwargs = kwargs
        self.max_iter = kwargs['max_iter']
        self.subsample_size = kwargs['subsample_size']
        self.mp = True if "mp" in kwargs.keys() and kwargs['mp'] else False
        self.classes = None

    def fit_single(self, K, y):
        for idx, i in enumerate(self.classes):
            y_binary = np.where(y == i, 1, -1)
            model = ASKFSVMBinary(max_iter=self.max_iter, subsample_size=self.subsample_size)
            model.fit(K, y_binary)
            self.models[i] = model
            self.class_map[idx + 1] = i  # 1 maps to first class, -1 maps to rest

    def fit_multi(self, K, y):
        if MP_LIB == "ray":
            print("MP enabled using ray library")
            # init ray
            ray.init()
            # cast K to 3d ndarray
            K = np.stack(K, axis=0)
            # push K to shared object storage
            data_id = ray.put(K)

            rays = list()
            for idx, i in enumerate(self.classes):
                rays.append(fit_remote.remote(data_id, y, idx, i, self.max_iter, self.subsample_size))
            # wait until all ray jobs finished
            res = ray.get(rays)

            # fill the class variables
            for entry in res:
                self.models[entry[0]] = entry[2]
                self.class_map[entry[1] + 1] = entry[0]  # 1 maps to first class, -1 maps to rest

            # shutdown ray
            ray.shutdown()
        elif MP_LIB == "joblib":
            print("MP enabled using joblib library")
            # cast K to 3d ndarray
            K = np.stack(K, axis=0)
            shape = K.shape
            dtype = str(K.dtype)

            # Write data to shared memory and clear K afterwards to free memory
            shm = shared_memory.SharedMemory(name='shared_ndarray', create=True, size=K.nbytes)
            data_sm = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            data_sm[:] = K[:]
            del K
            _ = gc.collect()

            # start parallel jobs
            res = Parallel(n_jobs=-1)(delayed(fit_shared)(y, idx, i, self.max_iter, self.subsample_size, shape, dtype)
                                              for idx, i in enumerate(self.classes))

            # fill the class variables
            for entry in res:
                self.models[entry[0]] = entry[2]
                self.class_map[entry[1] + 1] = entry[0]  # 1 maps to first class, -1 maps to rest

            shm.close()
            shm.unlink()

    def fit(self, K, y):
        start = timer()
        self.classes = np.unique(y)
        if self.mp:
            self.fit_multi(K, y)
        else:
            self.fit_single(K, y)
        stop = timer()
        diff = stop - start
        print("OneVsRestClassifier fit took " + str(diff))


    def predict(self, K):
        predictions = []
        score_matrix = np.zeros(shape=(len(K[0]),len(self.models)))
        for class_i, model in self.models.items():
            scores = model.decision_function(K)
            score_matrix[:, class_i] = scores
        pause = "pause"
        predictions = np.argmax(score_matrix, axis=1)

        return predictions