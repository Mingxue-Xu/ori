from ori.data_loader.loader import oi_loader
from ori.partition.partitioner import oi_partitioner
from ori.data_loader.util import *
from ori.train.target import *
from ori.train.infer import oi_infer
from ori.inspect.bias import oi_bias
import yaml
with open("config/oi_user_tiny.yaml", 'r') as stream:
    cfg = yaml.safe_load(stream)

data_loader = oi_loader(cfg['raw'].format(cfg['setting']))
data_loader.preprocess()
data_loader.set_labels()
data_loader.copy_image()

partitioner = oi_partitioner(cfg['meta'].format(cfg['setting']))
partitioner.load_data()
partitioner.split_metaset()
partitioner.split_P()
partitioner.save()

target_model = oi_target(cfg['target'].format(cfg['setting']))
target_model.build_model()
target_model.load_data()
target_model.compile()
target_model.train()

proxy_model = oi_target(cfg['proxy'].format(cfg['setting']))
proxy_model.build_model()
proxy_model.load_data()
proxy_model.compile()
proxy_model.train()

infer = oi_infer(cfg['infer'].format(cfg['setting']))
infer.load_all()
layers=infer.cfg['layers_list']
infer.load_model()
for l in layers:
    infer.cfg['layer'] = l
    infer.get_data(to_save=True, layer=l)
    infer.train()






