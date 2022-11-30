from .base import base_partitioner
import yaml
import pandas as pd
import numpy as np
import sympy
import random
from .util import *
class oi_partitioner(base_partitioner):
    def __init__(self,cfg_file):
        print("\nInit OpenImage Partitioner ...")
        with open(cfg_file, 'r') as stream:
            self.cfg = yaml.safe_load(stream)

    def load_data(self):
        self.data = pd.read_csv(self.cfg['data'])
        self.data_t = pd.read_csv(self.cfg['data_t'])
        self.eqset = self.data_t[self.cfg['eqset']].unique()
        print("The amount of selected eqsets - {}".format(len(set(self.eqset))))
        print(self.data.columns)
        self.eqset = np.asarray(random.choices(self.eqset, k = len(self.eqset)))
        
    def split_metaset(self):
        num_id = len(self.eqset)
        sum_t = self.cfg['P']['target'] + self.cfg['P']['proxy'] + self.cfg['P']['extra']
        alpha = num_id/sum_t

        self.inter_l = {'target': range(np.int(alpha*self.cfg['P']['target'])),
                        'proxy': range(np.int(alpha*self.cfg['P']['target']), 
                                  np.int(alpha*(self.cfg['P']['target']+self.cfg['P']['proxy']))),
                        'extra': range(np.int(alpha*(self.cfg['P']['target']+self.cfg['P']['proxy'])), 
                                  np.int(alpha*(self.cfg['P']['target']+self.cfg['P']['proxy']+self.cfg['P']['extra'])))
        }
        self.target_eqset = self.eqset[self.inter_l['target']]
        self.proxy_eqset = self.eqset[self.inter_l['proxy']]
        self.extra_eqset = self.eqset[self.inter_l['extra']]

        print("The amount of eqsets - target:{}, proxy:{}, extra:{}".format(len(set(self.target_eqset)),
                                                          len(set(self.proxy_eqset)),
                                                          len(set(self.extra_eqset))))

    def split_P(self):
        self.intra_r = {}
        self.intra_l = {}
        self.dataset = {"target":{}, "proxy":{}, "extra":{}}
        def intra_split(k, eqset_list):
            data_l = {}
            for t in eqset_list:    # for each eqset
                data_l[0], data_l[1] = split_list(list_t = self.data.loc[self.data[self.cfg['eqset']]==t].index,
                                    ratio=self.intra_r[k])

                for k_t in self.cfg[k][:2]:
                    idx = self.cfg[k].index(k_t)
                    data_t = self.data.iloc[data_l[idx]]
                    if k_t not in self.dataset[k].keys():
                        self.dataset[k][k_t]=data_t
                    else:
                        self.dataset[k][k_t]=pd.concat([self.dataset[k][k_t], data_t],ignore_index=True)

        for k in ['target', 'proxy']:
            self.intra_r[k] = float(sympy.Rational(self.cfg['intra'][k]))

            if k == 'proxy':    # meta_pos/meta_neg split
                eqset_t = {}
                eqset_t['nonmem'], eqset_t['mem'] = split_list(list_t=list(self.proxy_eqset),
                                           ratio=self.cfg['inter'][k])
                self.proxy_eqset = eqset_t
                intra_split(k=k, eqset_list=list(self.proxy_eqset['mem']))
            elif k == 'target':
                intra_split(k=k, eqset_list=list(self.target_eqset))

        self.dataset['proxy'][self.cfg['proxy'][-1]] = self.data[self.data[self.cfg['eqset']].isin(self.proxy_eqset['nonmem'])]
        self.dataset['extra'][self.cfg['extra'][-1]] = self.data[self.data[self.cfg['eqset']].isin(self.extra_eqset)]
        
    def save(self): 
        for k in  ["target", "proxy", "extra"]:
            for i in self.cfg[k]:
                self.dataset[k][i].to_csv(self.cfg['files'][k][i], index=False)
                print("{} dataset shape - {}".format(self.cfg['files'][k][i], self.dataset[k][i].shape))
            print("{} dataset shape - {}".format(k, count_dict_elements(self.dataset[k])))


