from .base import *
import yaml
import datetime
import tensorflow_addons as tfa
from .util import *

tf.get_logger().setLevel('ERROR')

class oi_target(target_base):
    def __init__(self,cfg_file) -> None:
        print("Init OpenImage Target ...")
        with open(cfg_file, 'r') as stream:
            self.cfg = yaml.safe_load(stream)
        limit_gpu()

    def build_model(self):
        self.model = tf.keras.applications.MobileNetV2(
            input_shape=None, alpha=1.0, include_top=True, weights=None,
            input_tensor=None, pooling=None, classes=len(self.cfg['class_idx']))

    def load_data(self):
        self.ds={}
        for l in ['train_set', 'test_set']:
            self.ds[l] = oi_datasets(
                cfg = {
                    'root': self.cfg['root'],
                    'csv_path': self.cfg[l],
                    'batch_size': self.cfg['batch_size']
                     }
            )
            self.ds[l] = self.ds[l].build_oi_dataset(class_idx=self.cfg['class_idx'])

    def evaluate(self):
        results = self.model.evaluate(self.ds['test_set'],
                                      batch_size= self.cfg['batch_size'])
        print("test loss, test acc:", results)

    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(self.cfg['lr'])),
            loss=self.cfg['loss'],
            metrics= [tfa.metrics.F1Score(num_classes=self.cfg['num_class'], threshold=0.5,
                                         average='macro') if len(self.cfg['class_idx']) > 1
                      else self.cfg['class_idx']]
        )
    def train(self):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        checkpoint_filepath = 'checkpoint'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        self.model.fit(self.ds['train_set'],
                       epochs=self.cfg['epochs'],
                       callbacks=[model_checkpoint_callback, tensorboard_callback],
                       shuffle=True)
        self.model.save(self.cfg['save_dir'])

    def load_model(self):
        self.model.load_weights(self.cfg['save_dir'])
