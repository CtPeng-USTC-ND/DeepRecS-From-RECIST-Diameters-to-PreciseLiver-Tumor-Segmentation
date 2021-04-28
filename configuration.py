
aug_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
no_aug_args = dict(rotation_range=0,
                    width_shift_range=0,
                    height_shift_range=0,
                    shear_range=0,
                    zoom_range=0,
                    horizontal_flip=False,
                    fill_mode='nearest')