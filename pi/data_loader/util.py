import os.path
import shutil
import pandas as pd
import numpy as np

def find_substr_cols(list_t, str_t):
    res = []
    for str_tt in list_t:
        if str_t in str_tt:
            res.append(str_tt)
    return res

def sum_cols(df, col_list, user_id, friends, review_col):
    df_add = df[[user_id,friends,review_col]]
    df_t = df[col_list]
    df_t['col_sum'] = df_t.apply(lambda x: x.sum(), axis=1)
    df_t=pd.concat([df_add, df_t], axis=1)
    return df_t

def two_list_sub(l1,l2):
    res=np.zeros(len(l2))
    set_l2=set(l2)
    l1=list(set(l1))
    for i in range(len(l1)):
        if l1[i] in set_l2:
            res[l2.index(l1[i])]=1.0
    return res

def search_and_copy(filename, path, target_dir, dir_separated=False):
    dir_t=filename[0]
    if dir_separated:
        path_list=["train_{}".format(dir_t),
                   "validation", "test"]
    else:
        path_list=[""]
    for file_t in path_list:
        if os.path.exists(path+ file_t+"/{}.jpg".format(filename)):
            try:
                shutil.copy(path+ file_t+"/{}.jpg".format(filename), target_dir)
                return True
            except:
                print("Error when copying files:" + path+ file_t+"/{}.jpg".format(filename))
                return False




