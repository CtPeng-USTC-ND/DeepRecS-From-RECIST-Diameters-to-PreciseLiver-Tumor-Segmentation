from liver_tumor_segmentation.CGBS_Net import *
from liver_tumor_segmentation.loss import *
from keras.optimizers import *
from liver_tumor_segmentation.CGBS_data_generator import  *
from keras.callbacks import *
import os
from keras.callbacks import ReduceLROnPlateau
from keras import losses
from configuration import *

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr
def train():
    batch_size = 4 #4 for single GPU; 8 for two GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    trainGene = trainGenerator(batch_size, data_path='/data',
                               folder='train', aug_dict=aug_args, seed = 1, interaction='RECIST')
    devGene = trainGenerator(batch_size, data_path='/data',
                             folder='dev', aug_dict=no_aug_args, seed = 1, interaction='RECIST')
    testGene = testGenerator(test_path='test_path', interaction='RECIST')

    model = CGBS_Net(input_shape=(256, 256, 4),rate=3)
    model.summary()

    # GPU_COUNT = 2
    # model = multi_gpu_model(original_model, GPU_COUNT)

    opt=SGD(lr=4e-4, decay=1e-6, momentum=0.9, nesterov=True)
    lr_metric = get_lr_metric(opt)
    model.compile(optimizer=opt, loss={'out_seg': dice_coef_loss, 'out_shape': losses.binary_crossentropy},
                  loss_weights={'out_seg': 1, 'out_shape': 1}, metrics=[dice_coef, lr_metric])

    csv_logger = CSVLogger('./Models/'+'CGBS_Net.csv', append=True)  # ss-0.01
    # tensorboard = TensorBoard(log_dir='./tmp/graph', write_graph=True, write_images=True)
    # earlystopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')

    model_checkpoint = ModelCheckpoint(
        './Models/CGBS/{epoch:02d}-{val_out_seg_dice_coef:.4f}.h5',
        monitor='val_out_seg_loss',
        verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_out_seg_loss', factor=0.1, patience=50, mode='auto')
    model.fit_generator(generator=trainGene, steps_per_epoch=int(5000/batch_size),
                        epochs=500, validation_data=devGene,
                        validation_steps=int(5000/batch_size), verbose=2,
                        callbacks=[model_checkpoint, csv_logger, reduce_lr])


train()

