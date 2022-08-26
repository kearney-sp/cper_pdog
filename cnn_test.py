import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

n_cpu = os.cpu_count()
print('CPUs: ', n_cpu)

def memory_usage():
    """Memory usage of the current process in GB."""
    status = None
    result = {'peak': 0, 'rss': 0}
    try:
        # This will only work on systems with a /proc file system
        # (like Linux).
        status = open('/proc/self/status')
        for line in status:
            parts = line.split()
            key = parts[0][2:-1].lower()
            if key in result:
                result[key] = int(parts[1])*0.000001
    finally:
        if status is not None:
            status.close()
    return result

memory_usage()

### Prep lists of input files

from glob import glob
import re
import pandas as pd
from random import sample

# set working directory
os.chdir('/project/cper_neon_aop/cper_pdog_uas')

# set directories for training data and labels
DATA_FOLDER = './cnn_train_images/{}_rgb.tif'
LABEL_FOLDER = './cnn_train_labels/{}_labels.tif'

# read in csvs with training information
df_tiles = pd.read_csv('train_tiles/train_bboxes_all_assigned.csv')
df_polys = pd.read_csv('train_polys/train_polys_all.csv')

# get all ids to be used
label_files = glob(LABEL_FOLDER.replace('{}', '*'))
all_ids = [re.sub('_labels.tif', '', os.path.basename(f)) for f in label_files]
all_tiles = list(set(['_'.join(y.split('_')[-3:]) for y in all_ids]))

# separate training and test data and get paths to files
all_files = glob(DATA_FOLDER.replace('{}', '*'))
all_train_tiles = [x for x in df_tiles.apply(lambda x: '_'.join([x.Pasture, x.Tile]) if x.Train == 1 else '', axis=1) if x != '' and x in all_tiles]
test_tiles = list(set(all_tiles) - set(all_train_tiles))

all_train_ids = [x for x in all_ids if '_'.join(x.split('_')[-3:]) in all_train_tiles]
test_ids = list(set(all_ids) - set(all_train_ids))

valid_ids = sample(all_train_ids, int(np.ceil(len(all_train_ids)*0.3)))
train_ids = list(set(all_train_ids) - set(valid_ids))

train_files = [f for f in all_files if '_'.join(os.path.basename(f).split('_')[:-1]) in train_ids]
valid_files = [f for f in all_files if '_'.join(os.path.basename(f).split('_')[:-1]) in valid_ids]
test_files = [f for f in all_files if '_'.join(os.path.basename(f).split('_')[:-1]) in test_ids]

#df_tiles

### Dataloader
###Writing helper class for data extraction, tranformation and preprocessing

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from skimage import io
if __name__ == '__main__':
    class Dataset(BaseDataset):
        """Read images, apply augmentation and preprocessing transformations.

        Args:
            ids (list): list of unique ids for all images
            images_path (str): path to data images
            masks_path (str): path to label masks
            class_values (list): values of classes to extract from segmentation mask
            augmentation (albumentations.Compose): data transfromation pipeline 
                (e.g. flip, scale, etc.)
            preprocessing (albumentations.Compose): data preprocessing 
                (e.g. noralization, shape manipulation, etc.)

        """

        CLASSES = ['other', 'burrow']

        def __init__(
                self, 
                ids,
                images_path, 
                masks_path, 
                classes=None, 
                augmentation=None, 
                preprocessing=None,
        ):
            # get IDs as attribute
            self.ids = ids

            # List of files
            self.images_fps = [images_path.format(id) for id in ids]
            self.masks_fps = [masks_path.format(id) for id in ids]

            # convert str names to class values on masks
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

            self.augmentation = augmentation
            self.preprocessing = preprocessing

        def __getitem__(self, i):

            # read data
            image = np.asarray(io.imread(self.images_fps[i]), dtype='uint8')
            mask = np.asarray(io.imread(self.masks_fps[i]), dtype='uint8')
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #mask = cv2.imread(self.masks_fps[i], 0)

            # extract certain classes from mask (e.g. cars)
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1)#.astype('float32')
            #print('fetched: ', self.ids[i])
            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

                #print(memory_usage())
            return image, mask

        def __len__(self):
            return len(self.ids)

    # Look at a training image with a burrow

    dataset = Dataset(train_ids, DATA_FOLDER, LABEL_FOLDER, classes=['burrow'])

    image, mask = dataset[5] # get sample with known burrow
    visualize(
        image=image, 
        burrow_mask=mask.squeeze(),
    )

    ### Augmentations

    import albumentations as albu

    def get_training_augmentation():
        train_transform = [

            albu.HorizontalFlip(p=0.5),

            #albu.ShiftScaleRotate(scale_limit=0.0, rotate_limit=45, shift_limit=0.1, p=1, border_mode=0),

            #albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
            #albu.RandomCrop(height=320, width=320, always_apply=True),

            albu.GaussNoise(p=0.2, var_limit=1.0),
            #albu.Perspective(p=0.5),

            albu.OneOf(
                [
                    #albu.CLAHE(p=1), # required int8 images
                    albu.RandomBrightnessContrast(p=1),
                    albu.RandomGamma(p=1),
                    albu.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.Sharpen(p=1),
                    albu.Blur(blur_limit=(3, 7), p=1),
                    albu.MotionBlur(blur_limit=(3, 7), p=1),
                ],
                p=0.9,
            ),
        ]
        return albu.Compose(train_transform)


    def get_validation_augmentation():
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
            albu.PadIfNeeded(384, 480)
        ]
        return albu.Compose(test_transform)


    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')


    def get_preprocessing(preprocessing_fn):
        """Construct preprocessing transform

        Args:
            preprocessing_fn (callbale): data normalization function 
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose

        """

        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
        return albu.Compose(_transform)

    def simp_aug():
        train_transform = [
            #albu.Perspective(p=0.5),
        ]
        return albu.Compose(train_transform)

    simp_aug_dataset = Dataset(
        train_ids,
        DATA_FOLDER,
        LABEL_FOLDER,
        augmentation=simp_aug(),
        classes=['burrow'])

    image, mask = simp_aug_dataset[5] # get sample with known burrow
    visualize(
        image=image, 
        burrow_mask=mask.squeeze(),
    )

    #valid_ids.index('poly_48_22W_random_8')

    #### Visualize resulted augmented images and masks

    augmented_dataset = Dataset(
        valid_ids,
        DATA_FOLDER,
        LABEL_FOLDER,
        augmentation=get_training_augmentation(),
        classes=['burrow'])

    # same image with different random transforms
    for i in range(3):
        image, mask = augmented_dataset[68]
        visualize(image=image, mask=mask.squeeze(-1))

    ### Create and train model

    import torch
    import numpy as np
    import segmentation_models_pytorch as smp

    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['burrow']
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = Dataset(
        train_ids,
        DATA_FOLDER,
        LABEL_FOLDER,
        #augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES)

    valid_dataset = Dataset(
        valid_ids,
        DATA_FOLDER,
        LABEL_FOLDER,
        #augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES)


    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    # train model for 5 epochs

    max_score = 0

    for i in range(0, 5):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './cnn_results/best_model.pth')
            print('Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')