from .base import *
import yaml
from .util import *
import pandas as pd
import numpy as np
import os
from keras import backend as K
import tensorflow as tf
import pickle

class oi_infer(infer_base):
    def __init__(self, cfg_file):
        with open(cfg_file, 'r') as stream:
            self.cfg = yaml.safe_load(stream)
        limit_gpu()
        self.raw = {}
        self.data = {}
        self.dataset = {}

    def load_raw(self, k):
        self.raw[k] = {}
        df = pd.read_csv(self.cfg['data'][k])
        eqset_list = set(df[self.cfg['eqset']])
        for t in eqset_list:
            self.raw[k][t]= df.loc[df[self.cfg['eqset']]==t]
            self.raw[k][t] = oi_datasets(cfg = {
                    'root': self.cfg['root'],
                    'batch_size': None
                     },
            df=self.raw[k][t])
            self.raw[k][t]=self.raw[k][t].build_oi_dataset()
    def load_all(self):
        for k in self.cfg['data'].keys():
            self.load_raw(k)

    def gen_data(self, k, model):
        if not isinstance(model, list):
            outputs = [layer.output for layer in model.layers]
            functons = [K.function([model.input], [out]) for out in outputs]

            key_list=list(self.raw[k].keys())
            key_list.reverse()
            for t in key_list:
                str_file = self.cfg['layer_file'].format(k, t.split("/")[-2])
                data_t = tf.data.experimental.get_single_element(self.raw[k][t])
                for i in self.cfg['layers_list']:
                    str_path = self.cfg['layer_dir'].format(i)

                    if data_t[0].shape[0]>1000:
                        data_l = np.array_split(data_t[0], data_t[0].shape[0]/1000+1)
                        layer_outs = [functons[i](data_tt)[0] for data_tt in data_l ]
                        layer_outs = np.vstack(layer_outs)
                    else:
                        layer_outs = functons[i]([data_t[0]])[0]

                    if not os.path.exists(str_path):
                        os.makedirs(str_path)

                    np.save(str_path + str_file, layer_outs)
                    del layer_outs
        else:
            functions = []
            for m in model:
                functions.append(build_layer_func(model=m, layers=self.cfg['layers_list']))
            key_list = list(self.raw[k].keys())
            layer_name={}
            for i in self.cfg['layers_list']:
                layer_name[i]=model[0].layers[i].name

            for t in key_list:
                str_file = self.cfg['layer_file'].format(k, t.split("/")[-2])
                data_t = tf.data.experimental.get_single_element(self.raw[k][t])
                for i in self.cfg['layers_list']:
                    str_path = self.cfg['layer_dir'].format(i)
                    if data_t[0].shape[0] > 1000:
                        layer_outs=[]
                        for j in range(len(functions)):
                            data_l = np.array_split(data_t[0], data_t[0].shape[0] / 1000 + 1)
                            layer_o = [functions[j][layer_name[i]](data_tt)[0] for data_tt in data_l]
                            layer_o = np.vstack(layer_o)
                            layer_outs.append(layer_o)
                    else:
                        layer_outs = []
                        for j in range(len(functions)):
                            layer_o = functions[j][layer_name[i]]([data_t[0]])[0]
                            layer_outs.append(layer_o)
                        
                    flag=False
                    for k1 in range(len(model)):
                        for k2 in range(len(model)):
                            if not np.array_equal(a1=layer_outs[k1], a2=layer_outs[k2]):
                                print("{} and {} of {} are the different".format(k1, k2, t))
                                flag=True

                    if not flag:
                        continue

                    if not os.path.exists(str_path):
                        os.makedirs(str_path)

                    np.save(str_path + str_file, layer_outs)
                    del layer_outs

    def load_model(self, is_print=False):
            if isinstance(self.cfg['proxy'], list):
                self.proxy = []
                for m in self.cfg['proxy']:
                    self.proxy.append(tf.keras.models.load_model(m))
            else:
                self.proxy = tf.keras.models.load_model(self.cfg['proxy'])
            self.target = tf.keras.models.load_model(self.cfg['target'])
            if is_print == True:
                self.proxy.summary()

    def get_data(self, to_save=True, layer=155):
        if to_save:
            for k in self.cfg['data'].keys():
                if "train" in k:
                    self.gen_data(k=k, model=self.proxy)
                elif "test" in k:
                    self.gen_data(k=k, model=self.target)
        else:
            self.data={"train_x":[], "test_x":[], "train_y":[], "test_y":[]}
            str_path = self.cfg['layer_dir'].format(layer)
            for k in self.cfg['data'].keys():
                for t in self.raw[k].keys():
                    str_file = self.cfg['layer_file'].format(k, t.split("/")[-2])
                    data_t = np.squeeze(np.load(str_path+str_file, allow_pickle=True))

                    if 'train' in k:
                        shape_t = list(data_t.shape)[:(-1) * len(self.cfg['layer_shape'][layer])]
                        if len(shape_t) > 1:
                            data_t = np.array_split(data_t, shape_t[0], axis=0)
                            for i in range(len(data_t)):
                                self.data['train_x'].append(data_t[i])
                                self.data['train_y'].append(1 if 'pos' in k else 0)
                        else:
                            self.data['train_x'].append(data_t)
                            self.data['train_y'].append(1 if 'pos' in k else 0)

                    elif 'test' in k:
                        if isinstance(data_t, list):
                            for i in range(len(data_t)):
                                self.data['test_x'].append(data_t[i])
                                self.data['test_y'].append(1 if 'pos' in k else 0)
                        else:
                            self.data['test_x'].append(data_t)
                            self.data['test_y'].append(1 if 'pos' in k else 0)
            # reshape
            for k in ['train_x', 'test_x']:
                max_shape_length=0
                min_shape_length=1000
                for t in self.data[k]:
                    t_shape=list(t.shape)
                    if max_shape_length<len(t_shape):
                        max_shape_length=len(t_shape)
                    elif min_shape_length>len(t_shape):
                        min_shape_length=len(t_shape)
                if max_shape_length!=min_shape_length:
                    for i in range(len(self.data[k])):
                        t=self.data[k][i]
                        if len(t.shape)>min_shape_length:
                            t_shape=list(t.shape)
                            t=t.reshape(([-1]+t_shape[2:]))
                            self.data[k][i] = t
    def featurization(self):
        self.dataset={'train_x':[], 'test_x':[]}
        for k in ['train_x', 'test_x']:
            for i in range(len(self.data[k])):
                featurizer=tiny_featurizer(name=self.cfg['feature'],  kwargs_t=self.cfg['feat_args'])
                featurizer.load_data(data=self.data[k][i])
                self.dataset[k].append(featurizer.process())

    def save_res(self, filename, best_score, best_parameters, result_report, bag_size, feat_args):
        df = pd.DataFrame()
        if os.path.exists(filename):
            df = pd.read_csv(filename, index_col=0)

        df_t = {}
        for col in self.cfg["log_cols"]["from_yaml"]:
            df_t[col] = self.cfg[col]
        df_t["feat_args"] = feat_args
        df_t["bag_size"] = bag_size
        df_t["best_score"] = best_score
        df_t["params"] = best_parameters
        df_t["report"] = result_report

        if df.size != 0:
            df = df.append(df_t, ignore_index=True)
        else:
            df = pd.DataFrame({key: pd.Series(value) for key, value in df_t.items()})

        df.to_csv(filename)

    def train(self):
        self.get_data(to_save=False, layer=self.cfg['layer'])
        self.featurization()
        _, best_score, best_parameters, result_report = meta_train(train_x=self.dataset["train_x"],
                                                                   train_y=self.data["train_y"],
                                                                   test_x=self.dataset["test_x"],
                                                                   test_y=self.data["test_y"],
                                                                   name=self.cfg["classifier"],
                                                                   args=self.cfg["args"])
        self.save_res(filename=self.cfg['log'].format("MobileNet"),
                      best_score=best_score,
                      best_parameters=json.dumps(best_parameters),
                      result_report=result_report,
                      bag_size=np.nan,
                      feat_args=self.cfg['feat_args'])

        self.meta_model = _
        pickle.dump(self.meta_model, open(self.cfg['meta_model'].format(self.cfg['feature'], self.cfg['set_name'], self.cfg['layer']), 'wb'))

























            