from liver_tumor_segmentation.CGBS_Net import *
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
def test():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    model = CGBS_Net(input_shape=(256, 256, 4),rate=2)
    model.load_weights('/weights.h5',
                       by_name=True)

    opt=SGD(lr=4e-4, decay=1e-6, momentum=0.9, nesterov=True)
    lr_metric = get_lr_metric(opt)
    model.compile(optimizer=opt, loss={'out_seg': dice_coef_loss, 'out_shape': binary_crossentropy},
                  loss_weights={'out_seg': 1, 'out_shape': 1}, metrics=[dice_coef, lr_metric])#0.01, 0.001, 0.0005，

    results = model.predict_generator(testGene, 1000)
    save_path = './results'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saveResult_2out(save_path, results)

# if __name__ == 'main':
test()
