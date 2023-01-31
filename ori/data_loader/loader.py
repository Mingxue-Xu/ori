from .base import base_loader
import yaml, os
from .util import *
class oi_loader(base_loader):
    def __init__(self,cfg_file):
        print("\nInit OpenImage Dataloader ...")
        with open(cfg_file, 'r') as stream:
            self.cfg = yaml.safe_load(stream)

    def connect_tables(self, key="label"):
        self.df=pd.DataFrame([])
        if key=="label":
            file_list=self.cfg['label']
        elif key=="user":
            file_list = self.cfg['user']
        elif key=="multi-label":
            file_list = self.cfg['multi-label']

        for file in file_list:
            if self.df.size==0:
                self.df=pd.read_csv(self.cfg['root'] + file)
            else:
                self.df = pd.concat([self.df, pd.read_csv(self.cfg['root'] + file)])
        print("{} total samples: {}".format(key, self.df.shape))

    def find_class_id(self, key_words=None):
        df_class = pd.read_csv(self.cfg["root"]+self.cfg["class_description"])
        df_class = df_class.apply(lambda x: x.str.lower())
        column = list(df_class.columns)
        self.class_id = []

        if key_words==None:
            key_words=self.cfg["key_words"]

        for cls in key_words:
            res = list((df_class.loc[lambda df: df[column[1]].str.contains(cls)])[column[0]])
            print("keyword {} contains {}".format(cls, res))
            self.class_id.extend(res)

    def select_imgs(self, df):
        self.data = pd.DataFrame([])
        for cls in self.class_id:
            df_t = df.loc[lambda df: df['LabelName'] == cls]
            df_t = df_t.loc[lambda d_t: d_t['Confidence'] == 1]
            if self.data.size==0:
                self.data = df_t
            else:
                self.data = pd.concat([self.data, df_t])
        print("Selected images - {}".format(self.data.shape))

    def select_users(self, data_list, df, limit=None, user_only=False):
        self.users = df.loc[df[self.cfg['data']].isin(data_list)]
        print("Original Selected users - {}".format(self.users.shape))
        df_count = self.users.groupby([self.cfg['key']], as_index=False)[self.cfg['key']].agg({'cnt': 'count'})
        if limit==None:
            limit=self.cfg['limit']

        self.users = (df_count.loc[df_count['cnt'] >= limit])
        print("Selected users - {}".format(self.users.shape))

        if user_only:
            return

        self.df = self.df.loc[self.df[self.cfg['key']].isin(list(self.users[self.cfg['key']]))]
        self.df = self.df.loc[self.df[self.cfg['data']].isin(list(self.data[self.cfg['data']]))]

        print("Total data - {}".format(self.df.shape))

    def save(self):
        self.df.to_csv(self.cfg['save']['filename'].format(self.cfg["setting"]), index=False)

    def name_to_cls_label(self, name_list):
        label_df = pd.read_csv(self.cfg["root"] + self.cfg["class_description"])
        res=[]
        for n in name_list:
            res.append(list(label_df.loc[label_df[self.cfg['label_name']]==n][self.cfg['label_col']])[0])
        return res

    def set_labels(self):
        self.data = pd.read_csv(self.cfg['save']['filename'].format(self.cfg["setting"]))
        self.connect_tables(key="multi-label")
        self.df = self.df[[self.cfg['conf'], self.cfg['data'], self.cfg['label_col']]]
        img_list = list(set(list(self.data[self.cfg['data']])))
        label_list = list(self.name_to_cls_label(self.cfg['multi_label_key']))
        multi_label  ={}
        for img in img_list:
            img_label = list((self.df.loc[self.df[self.cfg['data']]==img])[self.cfg['label_col']])
            multi_label[img]=two_list_sub(img_label,label_list)
        self.data['multi_label']=self.data[self.cfg['data']].map(multi_label)
        self.data = self.data[[self.cfg['key'], self.cfg['data'], self.cfg['label_task']]]
        self.data.to_csv(self.cfg['save']['label_file'].format(self.cfg["setting"]), index=False)

    def copy_image(self):
        self.data=pd.read_csv(self.cfg['save']['label_file'].format(self.cfg["setting"]))
        data_list=self.data[self.cfg['data']]
        for data in data_list:
            search_and_copy(data, path=self.cfg['root'], target_dir=self.cfg['target_path'].format(self.cfg["setting"]),
                            dir_separated=False)

    def preprocess(self):
        self.find_class_id()
        self.connect_tables(key="label")
        self.select_imgs(df=self.df)
        self.connect_tables(key="user")
        self.select_users(data_list=list(self.data[self.cfg['data']]), df=self.df)
        self.save()



















        
        
        
        

        
        
        
        
