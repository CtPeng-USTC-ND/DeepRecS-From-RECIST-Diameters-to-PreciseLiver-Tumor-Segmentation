from RECIST_mark_propagation.RMP_data_generator import *
from keras import models
from keras.callbacks import *
from keras.utils.generic_utils import CustomObjectScope
from RECIST_mark_propagation.RMP_Net import *
from keras.optimizers import SGD
from keras.losses import mean_squared_error
import os

train_path = "/data0/zy/db/medical/kpt256/train"
# test_path = "/data0/zy/db/medical/kpt256/test"
dev_path = "/data0/zy/db/medical/kpt256/test"
test_path = "/data0/zy/db/medical/kpt256/test/tumor_patch_000"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

csv_model_path = './Models/' + time.strftime('%Y-%m-%d_%H-%M/', time.localtime(time.time()))
if not os.path.exists(csv_model_path):
    os.makedirs(csv_model_path)
csv_logger = CSVLogger(time.strftime(csv_model_path + '%Y-%m-%d_%H-%M-%S.csv', time.localtime(time.time())))

model_saver =  ModelCheckpoint(csv_model_path + '{epoch:02d}-{mean_squared_error:.4f}-{val_mean_squared_error:.4f}.h5', monitor='val_mean_squared_error',
                               verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
# tensorboard = TensorBoard(log_dir='./tmp/log', histogram_freq=0, write_graph=True, write_images=False,
#                           embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

#earlystopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')

trainGene = trainGenerator(train_path=train_path, batch_size=20) #5
devGene = devGenerator(dev_path=dev_path, batch_size=20) #5
testGene = testGenerator(os.path.join(test_path,'CT'))
model = create_hourglass_network(num_classes=4, num_stacks=2,
                                 inres=(256,256), outres=(64,64),
                                 bottleneck=bottleneck_block)

model.compile(optimizer=SGD(lr=4e-4, decay=1e-6, momentum=0.9, nesterov=True),
              loss=mean_squared_error, metrics=['mse'])

model.fit_generator(generator=trainGene,verbose=1,
                    steps_per_epoch=5000, epochs=500,
                    validation_data=devGene, validation_steps=100,
                  callbacks=[model_saver, tensorboard, csv_logger])

