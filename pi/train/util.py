import sys, os, json
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

def limit_gpu():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def build_layer_func(model):
    inp=model.input
    layers=[layer for layer in model.layers]
    func_dict={}
    for i in range(len(layers)):
        layer_t=layers[i].output
        if isinstance(layer_t, dict):
            for k in layer_t.keys():
                func_dict[layers[i].name + "-" + k] = K.backend.function([inp], [layers[i].output[k]])
        else:
            func_dict[layers[i].name] = K.backend.function([inp], [layer_t])
    return func_dict

def get_layer_output(model=None, x=None, layers=[-1]):
    res = []
    for l in layers:
        if isinstance(l, int):
            layer = K.backend.function([model.input], [model.layers[l].output])
            res_t = layer([x])
        elif isinstance(l, str):
           layer = K.backend.function([model.input], [model.get_layer(l).output])
           res_t = layer([x])
        elif isinstance(l, dict):
            layer_name = list(l.keys()).pop()
            if isinstance(l[layer_name], str):
                layer = K.backend.function([model.input], [model.get_layer(layer_name).output])
                layer_output = layer([x])
                res_t= [[[layer_output[i][j][k] \
                    for i in range(len(layer_output))] \
                        for j in range(len(layer_output[0]))\
                            for k in l[layer_name]]]
            elif isinstance(l[layer_name], list):
                idx = l[layer_name][0]
                if not isinstance(idx, dict):
                    layer = K.backend.function([model.input], [model.get_layer(layer_name).output[idx]])
                else:
                    key_t = list(idx.keys())[0]
                    idx_t=idx[key_t][0]
                    layer = K.backend.function([model.input], [model.get_layer(layer_name).output[key_t][idx_t]])
                res_t = layer([x])
            res_t = np.squeeze((np.asarray(res_t)))
        res.append(res_t)
    return res

def meta_train(train_x, train_y, test_x, test_y, name, args):
    if name=="simple":
        methods=["knn","tree", "forest"]
        best_score=0
        best_parameters=None
        result_report=None
        model=None
        if isinstance(train_x, list):
            train_x=np.asarray(train_x)
            test_x = np.asarray(test_x)
        if len(train_x.shape) > 2:
            train_x = train_x.reshape(train_x.shape[0], -1)
            test_x = test_x.reshape(test_x.shape[0], -1)
        for m in methods:
            model_t, score, parameters, result_report_t=train(train_x, train_y, test_x, test_y, name=m, args=args)
            if score>best_score:
                model=model_t
                best_score=score
                best_parameters=parameters
                result_report=result_report_t
    else:
        model, best_score, best_parameters, result_report=train(train_x, train_y, test_x, test_y, name, args)
    return model, best_score, best_parameters, result_report

def train(train_x, train_y, test_x, test_y, name):
    def forest_train(train_x, train_y, test_x, test_y):
        best_score = 0
        best_model = None
        if isinstance(train_x, list):
            train_x = np.asarray(train_x)
            test_x = np.asarray(test_x)

        for max_depth in range(2, 100):
            model = RandomForestClassifier(random_state=0, max_depth=max_depth)
            model.fit(train_x, train_y)
            score = model.score(test_x, test_y)

            if score > best_score:
                best_score = score
                best_parameters = {'name': 'forest', 'max_depth': max_depth}
                best_model = model
        pred_y = model.predict(test_x)
        result_report = classification_report(pred_y, test_y)
        return best_model, best_score, best_parameters, result_report

    def knn_train(train_x, train_y, test_x, test_y):
        best_score = 0
        best_model = None
        if isinstance(train_x, list):
            train_x = np.asarray(train_x)
            test_x = np.asarray(test_x)
        limit_n = train_x.shape[0]
        for n_neighbors in range(2, limit_n if limit_n < 100 else 100):
            for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:

                model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
                model.fit(train_x, train_y)
                score = model.score(test_x, test_y)

                if score > best_score:
                    best_score = score
                    best_parameters = {'name': 'knn', 'n_neighbors': n_neighbors, 'algorithm': algorithm}
                    best_model = model

        pred_y = model.predict(test_x)
        result_report = classification_report(pred_y, test_y)
        return best_model, best_score, best_parameters, result_report

    def tree_train(train_x, train_y, test_x, test_y):
        best_score = 0
        best_model = None
        if isinstance(train_x, list):
            train_x = np.asarray(train_x)
            test_x = np.asarray(test_x)
        for criterion in ["gini", "entropy"]:
            for max_depth in range(2, 100):
                for splitter in ["best", "random"]:
                    model = tree.DecisionTreeClassifier(random_state=0, criterion=criterion, max_depth=max_depth,
                                                        splitter=splitter)
                    model.fit(train_x, train_y)
                    score = model.score(test_x, test_y)

                    if score > best_score:
                        best_score = score
                        best_parameters = {'name': 'tree', 'criterion': criterion, 'max_depth': max_depth,
                                           'splitter': splitter}
                        best_model = model
        pred_y = model.predict(test_x)
        result_report = classification_report(pred_y, test_y)
        return best_model, best_score, best_parameters, result_report

    def svm_train(train_x, train_y, test_x, test_y):

        best_score = 0
        para_list = [0.001, 0.01, 0.1, 0.1, 1, 10, 100, 1000, 10000]
        if isinstance(train_x, list):
            train_x = np.asarray(train_x)
            test_x = np.asarray(test_x)

        for gamma in para_list:
            for C in para_list:
                svm = SVC(gamma=gamma, C=C, probability=True)
                if len(train_x.shape) < 2:
                    train_x = train_x.reshape(-1, 1)
                    test_x = test_x.reshape(-1, 1)
                svm.fit(train_x, train_y)
                score = svm.score(test_x, test_y)
                if score > best_score:
                    best_score = score
                    best_parameters = {'name': 'svm', 'C': C, 'gamma': gamma}
        pred_y = svm.predict(test_x)
        result_report = classification_report(pred_y, test_y)

        return svm, best_score, best_parameters, result_report

    if name=="svm":
        return svm_train(train_x, train_y, test_x, test_y)
    if name=="knn":
        return knn_train(train_x, train_y, test_x, test_y)
    if name=="tree":
        return tree_train(train_x, train_y, test_x, test_y)
    if name=="forest":
        return forest_train(train_x, train_y, test_x, test_y)

def list_to_float(str_t):
    str_t = str_t[1:-2].split(" ")
    res = [float(ele) for ele in str_t]
    return res

def load_image(filename):
    img = Image.open(filename)
    img = img.convert('RGB')
    img.load()
    img = img.resize((224,224))
    data = np.asarray(img, dtype="int32")
    return data

class oi_datasets():
    def __init__(self, cfg=None, df=pd.DataFrame([])):
        self.cfg=cfg
        self.df=df

    def meta_process(self, root, csv_path=None, df=pd.DataFrame([]), class_idx=None):
        if self.df.empty:
            df = pd.read_csv(csv_path)
        file_name = df['ImageID'].tolist()
        self.path_list = []
        self.label = []
        self.label_t = df['multi_label'].tolist()
        for l in self.label_t:
            if class_idx==None:
                self.label.append(list_to_float(l))
            else:
                l=list_to_float(l)
                self.label.append([l[k] for k in class_idx])
        for file in file_name:
            self.path_list.append(root + "{}.jpg".format(file))

    def build_oi_dataset(self, class_idx=None):
        def load_image(img_path):
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)

            img = tf.image.resize(img, [224, 224])
            img /= 255.0
            return img
        if self.df.empty:
            self.meta_process(root=self.cfg['root'], csv_path=self.cfg['csv_path'], class_idx=class_idx)
        else:
            self.meta_process(root=self.cfg['root'], df=self.df, class_idx=class_idx)

        path_ds = tf.data.Dataset.from_tensor_slices(self.path_list)
        img_ds = path_ds.map(load_image)
        label_ds = tf.data.Dataset.from_tensor_slices(self.label)

        ds = tf.data.Dataset.zip((img_ds, label_ds))
        if self.cfg['batch_size']!=None:
            ds = ds.batch(self.cfg['batch_size'])
        else:
            ds = ds.batch(self.df.shape[0])

        return ds

class tiny_featurizer:
    def __init__(self, name, kwargs_t):
        self.name=name
        self.kwargs_t=kwargs_t
    def load_data(self, data):
        self.data=data
    def process(self, type='image'):
        if type == 'image':
            range_t = (0.0, 1.0)
        else:
            range_t = (-1.0, 1.0)

        if self.name == "histogram":
            res_t =  np.histogram(self.data, bins=self.kwargs_t['bins'], normed=False, range=range_t)[0]
            res_t = res_t/res_t.shape[0]

        elif self.name == "statics":
            res_t = np.concatenate(( np.max(self.data, axis=tuple(self.kwargs_t['axis'])).reshape(-1),
                                np.min(self.data, axis=tuple(self.kwargs_t['axis'])).reshape(-1),
                                np.mean(self.data, axis=tuple(self.kwargs_t['axis'])).reshape(-1),
                                np.percentile(a=self.data, axis=self.kwargs_t['axis'], q=20).reshape(-1),
                                np.percentile(a=self.data, axis=self.kwargs_t['axis'], q=25).reshape(-1),
                                np.percentile(a=self.data, axis=self.kwargs_t['axis'], q=40).reshape(-1),
                                np.percentile(a=self.data, axis=self.kwargs_t['axis'], q=50).reshape(-1),
                                np.percentile(a=self.data, axis=self.kwargs_t['axis'], q=60).reshape(-1),
                                np.percentile(a=self.data, axis=self.kwargs_t['axis'], q=75).reshape(-1),
                                np.percentile(a=self.data, axis=self.kwargs_t['axis'], q=80).reshape(-1),
                                np.var(self.data, axis=tuple(self.kwargs_t['axis'])).reshape(-1),
                                np.std(self.data, axis=tuple(self.kwargs_t['axis'])).reshape(-1)))

        return res_t

















