# from data_generator import *
from res_data_generator import *
from keras import models
from keras.callbacks import *
from keras.utils.generic_utils import CustomObjectScope
from SHN import *
from keras.optimizers import SGD
from keras.losses import mean_squared_error
from resnet import *
# from ohem import OHEM
from keras.backend.tensorflow_backend import set_session
import os
import tensorflow as tf

train_path = "/data0/zy/db/medical/kpt256/train"
# test_path = "/data0/zy/db/medical/kpt256/test"
dev_path = "/data0/zy/db/medical/kpt256/test"
test_path = "/data0/zy/db/medical/kpt256/test/tumor_patch_000"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = create_hourglass_network(num_classes=4, num_stacks=2,
                                 inres=(256,256), outres=(64,64),
                                 bottleneck=bottleneck_block)

model.compile(optimizer=SGD(lr=4e-4, decay=1e-6, momentum=0.9, nesterov=True),
              loss=mean_squared_error, metrics=['mse'])

results = model.predict_generator(testGene,1000, verbose=1)
save_path = os.path.join(test_path,'recist')
if not os.path.exists(save_path):
    os.makedirs(save_path)
saveResult(save_path,results)
