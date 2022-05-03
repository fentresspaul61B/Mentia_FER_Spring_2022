import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.BatchNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, ngf, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, ngf, 7),
                 nn.BatchNorm2d(ngf),
                 nn.ReLU()]

        # Downsampling
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.BatchNorm2d(out_features),
                      nn.ReLU()]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.BatchNorm2d(out_features),
                      nn.ReLU()]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, 3, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(3, ndf, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
                  nn.BatchNorm2d(ndf * 2),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(ndf * 4),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(ndf * 4, ndf * 8, 4, padding=1),
                  nn.InstanceNorm2d(ndf * 8),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(ndf * 8, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)





# from data_processor import X
# from data_processor import Y

# open('data_processor', X)
# print(X.shape)

"""--------------------------------------------------------------------------"""
# CNN NEURAL NET
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import random
random.seed(123)


batch_size  = 200
picture_size = 48
datagen_train  = ImageDataGenerator()
datagen_val = ImageDataGenerator()


folder_path = "/Users/paulfentress/Desktop/Mentia_Gans_Data.py/images_aged"

train_set = datagen_train.flow_from_directory(folder_path+"/train",
                                              target_size = (picture_size,picture_size),
                                              color_mode = "grayscale",
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=True)


test_set = datagen_val.flow_from_directory(folder_path+"/val",
                                              target_size = (picture_size,picture_size),
                                              color_mode = "grayscale",
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=False)



print("test_set type: " + str(type(test_set)))
print("Data Loading Complete")


"""Building the Neural Network from Video"""
no_of_classes = 6

model = Sequential()

#1st CNN layer
model.add(Conv2D(64,(3,3),padding = 'same',input_shape = (48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

#2nd CNN layer
model.add(Conv2D(128,(5,5),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout (0.25))

#3rd CNN layer
model.add(Conv2D(512,(3,3),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout (0.25))

#4th CNN layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

#Fully connected 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))


# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(no_of_classes, activation='softmax'))
opt = Adam(lr = 0.0001)
model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])


"""Training Model"""
checkpoint = ModelCheckpoint("./CNN_FER.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

early_stopping = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks_list = [early_stopping,checkpoint,reduce_learningrate]

epochs = 48

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])


print("Begining Model Training")
history = model.fit_generator(generator=train_set,
                                steps_per_epoch=train_set.n//train_set.batch_size,
                                epochs=epochs,
                                validation_data = test_set,
                                validation_steps = test_set.n//test_set.batch_size,
                                callbacks=callbacks_list
                                )

import pickle
model.save('CNN_FER.h5')
print("Model saved")
pickle.dump(history, open('cnn_history.pkl','wb'))
print("pickle complete.")


# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr'])
def plot_scores(history):
    scores = history.history["val_accuracy"]
    plt.plot(scores, '-x')
    plt.xlabel('epoch')
    plt.ylabel('CNN Model Validation Accuracy')
    plt.title('Validation Accuracy vs. No. of epochs');

plot_scores(history)
print("CNN Complete")



"""--------------------------------------------------------------------------"""


# TESS VGG NEURAL NET
#
# # importing the libraries
# import pandas as pd
# import numpy as np
#
# # for reading and displaying images
# from skimage.io import imread
# import matplotlib.pyplot as plt
#
#
# # for creating validation set
# from sklearn.model_selection import train_test_split
#
# # for evaluating the model
# from sklearn.metrics import accuracy_score
# from tqdm import tqdm
#
# # PyTorch libraries and modules
# import torch
# from torch.utils.data import random_split
# from torch.autograd import Variable
# from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
# from torch.optim import Adam, SGD
#
# # CNN NEURAL NET
# from keras.preprocessing.image import load_img, img_to_array
# from keras.preprocessing.image import ImageDataGenerator
#
#
# def ds_info(dataset):
#     dataset_size=len(dataset)
#     classes = dataset.classes
#     num_classes = len(classes)
#
#     #Initialize list
#     count = []
#     for x in range(num_classes):
#         count.append(0)
#
#     #Count every label
#     for x in range(dataset_size):
#         count[dataset[x][1]]+=1
#
#     print('Size of the dataset:' + str(dataset_size))
#     print('Number of classes:' + str(num_classes))
#     print('Samples for every class:')
#
#     #Print the count
#     for x in range(num_classes):
#         print(classes[x] + " : " + str(count[x]))
#
#
# #Transformations aplied to test DS.
# train_tf = tt.Compose([
#     tt.ColorJitter(brightness=0.2),
#     tt.Resize(size=(224,224)),
#     tt.RandomHorizontalFlip(),
#     tt.RandomRotation(5),
#     tt.ToTensor(),
# ])
#
# #Transformations aplied to test DS.
# test_tf= tt.Compose([
#     tt.Resize(size=(224,224)),
#     tt.ToTensor(),
# ])
# print("Data Sets")
#
#
# train_dir = "/content/images_aged copy/train"
# test_dir = "/content/images_aged copy/val"
#
# train_ds = ImageFolder(train_dir,train_tf)
# print("Training Data: " + str(len(train_ds)) + " images. ")
# test_ds = ImageFolder(test_dir,test_tf)
# print("Test Data: " + str(len(test_ds)) + " images. ")
#
#
# torch.manual_seed(123)
# val_size = 6000 #there are 29,782 train images and we want to use 20% for validation
# train_size = len(train_ds) - val_size
# train_ds, val_ds = random_split(train_ds, [train_size, val_size]) #Extracting the 6,000 validation images from the train set
#
# print("Validation Data: " + str(len(val_ds)) + " images. ")
#
#
# train_batch_size = 64
# val_batch_size = 32
# test_batch_size = 32
#
# train_dl = DataLoader(
#     train_ds,
#     batch_size=train_batch_size,
#     num_workers=2,
#     shuffle=True,
#     pin_memory=True
# )
#
# val_dl = DataLoader(
#     val_ds,
#     batch_size=val_batch_size,
#     num_workers=2,
#     shuffle=True,
#     pin_memory=True
# )
#
# test_dl = DataLoader(
#     test_ds,
#     batch_size=test_batch_size,
#     num_workers=2,
#     shuffle=False,
#     pin_memory=True
# )
#
#
# def get_default_device():
#     """Pick GPU if available, else CPU"""
#     if torch.cuda.is_available():
#         return torch.device('cuda')
#     else:
#         return torch.device('cpu')
#
# def to_device(data, device):
#     """Move tensor(s) to chosen device"""
#     if isinstance(data, (list,tuple)):
#         return [to_device(x, device) for x in data]
#     return data.to(device, non_blocking=True)
#
# class DeviceDataLoader():
#     """Wrap a dataloader to move data to a device"""
#     def __init__(self, dl, device):
#         self.dl = dl
#         self.device = device
#
#     def __iter__(self):
#         """Yield a batch of data after moving it to device"""
#         for b in self.dl:
#             yield to_device(b, self.device)
#
#     def __len__(self):
#         """Number of batches"""
#         return len(self.dl)
#
#
# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torch.optim as optim
# import time
# import torch.nn.functional as F
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from torchvision import models
#
#
# device = get_default_device() #Getting the device
# print(device)
#
# train_dl = DeviceDataLoader(train_dl, device) #Transfering train data to GPU
# val_dl = DeviceDataLoader(val_dl, device) #Transfering val data to GPU
# test_dl = DeviceDataLoader(test_dl, device)   #Transfering test data to GPU
#
#
# def accuracy(outputs, labels):
#     _, preds = torch.max(outputs, dim=1)
#     return torch.tensor(torch.sum(preds == labels).item() / len(preds))
#
#
#
# import random
# random.seed(123)
#
#
# # Look at the structure of VGG16
# vgg16 = models.vgg16(pretrained=True)
# vgg16.to(device)
# print(vgg16)
#
#
# class ImageClassificationBase(nn.Module):
#     def training_step(self, batch):
#         images, labels = batch
#         out = self(images)                  # Generate predictions
#         loss = F.cross_entropy(out, labels) # Calculate loss
#         return loss
#
#     def validation_step(self, batch):
#         images, labels = batch
#         out = self(images)                    # Generate predictions
#         loss = F.cross_entropy(out, labels)   # Calculate loss
#         acc = accuracy(out, labels)           # Calculate accuracy
#         return {'val_loss': loss.detach(), 'val_acc': acc}
#
#     def validation_epoch_end(self, outputs):
#         batch_losses = [x['val_loss'] for x in outputs]
#         epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
#         batch_accs = [x['val_acc'] for x in outputs]
#         epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
#         return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
#
#     def epoch_end(self, epoch, result):
#         print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
#             (epoch+1), result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
#
#
# class VGG_16(ImageClassificationBase):
#     def __init__(self, num_classes=6):
#       super().__init__()
#       # Use a pretrained model
#       self.network = models.vgg16(pretrained=True)
#       # Load the model features.
#       self.features = self.network.features
#       # Ave pool layer
#       self.avgpool = self.network.avgpool
#       # Define the FC layer.
#       # self.network.classifier[6].out_features = 6
#       self.fc = nn.Sequential(
#           nn.Linear(512*7*7, 1024),
#           nn.ReLU(),
#           nn.Linear(1024, 1024),
#           nn.ReLU(),
#           nn.Linear(1024, num_classes)
#         )
#
#     def forward(self, xb):
#       xb = self.features(xb)
#       xb = self.avgpool(xb)
#       xb = xb.view(xb.size(0),-1)
#       out = self.fc(xb)
#       return out
#
#     def freeze(self):
#         # To freeze the residual layers
#         for param in self.network.features.parameters():
#             param.require_grad = False
#
#     def unfreeze(self):
#         # Unfreeze all layers
#         for param in self.network.features.parameters():
#             param.require_grad = True
#
#
# from tqdm.notebook import tqdm
# @torch.no_grad()
# def evaluate(model, val_loader):
#     model.eval()
#     outputs = [model.validation_step(batch) for batch in val_loader]
#     return model.validation_epoch_end(outputs)
#
# def get_lr(optimizer):
#     for param_group in optimizer.param_groups:
#         return param_group['lr']
#
# def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
#                   weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
#     torch.cuda.empty_cache()
#     history = []
#
#     # Set up cutom optimizer with weight decay
#     optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
#     # Set up one-cycle learning rate scheduler
#     sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
#                                                 steps_per_epoch=len(train_loader))
#
#     for epoch in range(epochs):
#         # Training Phase
#         model.train()
#         train_losses = []
#         lrs = []
#         for batch in tqdm(train_loader):
#             loss = model.training_step(batch)
#             train_losses.append(loss)
#             loss.backward()
#
#             # Gradient clipping
#             if grad_clip:
#                 nn.utils.clip_grad_value_(model.parameters(), grad_clip)
#
#             optimizer.step()
#             optimizer.zero_grad()
#
#             # Record & update learning rate
#             lrs.append(get_lr(optimizer))
#             sched.step()
#
#         # Validation phase
#         result = evaluate(model, val_loader)
#         result['train_loss'] = torch.stack(train_losses).mean().item()
#         result['lrs'] = lrs
#         model.epoch_end(epoch, result)
#         history.append(result)
#     return history
#
#
# def plot_scores(history):
#     scores = [x['val_acc'] for x in history]
#     plt.plot(scores, '-x')
#     plt.xlabel('epoch')
#     plt.ylabel('Accuracy')
#     plt.title('VGG Validation Accuracy vs. No. of epochs');
#
#
# def plot_losses(history):
#     train_losses = [x.get('train_loss') for x in history]
#     val_losses = [x['val_loss'] for x in history]
#     plt.plot(train_losses, '-bx')
#     plt.plot(val_losses, '-rx')
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend(['Training', 'Validation'])
#     plt.title('Loss vs. No. of epochs');
#
#
# def show_sample(img, target):
#     plt.imshow(img.permute(1, 2, 0))
#     print('Labels:', target)
#
# def predict_image(img, model):
#     # Convert to a batch of 1
#     xb = to_device(img.unsqueeze(0), device)
#     # Get predictions from model
#     yb = model(xb)
#     # Pick index with highest probability
#     _, preds  = torch.max(yb, dim=1)
#     # Retrieve the class label
#     show_sample(img,test_ds.classes[preds[0].item()])
#
#
# model_vgg = to_device(VGG_16(), device)
# model_vgg
#
#
# model_vgg.freeze()
#
# epochs = 20
# max_lr = 0.001
# grad_clip = 0.1
# weight_decay = 1e-4
# opt_func = torch.optim.Adam
#
#
# historyVGG = fit_one_cycle(epochs, max_lr, model_vgg, train_dl, val_dl,
#                          grad_clip=grad_clip,
#                          weight_decay=weight_decay,
#                          opt_func=opt_func)
#
#
# PATH = "vgg_16_paul.h5"
# torch.save(model_vgg, PATH)
#
# plot_scores(historyVGG)
#
# pickle.dump(historyVGG, open('VGG16_history.pkl','wb'))
# print("VGG pickle complete.")
#
#
# import torch, gc
#
# gc.collect()
# torch.cuda.empty_cache()

# END OF TESS VGG MODEL.
"""--------------------------------------------------------------------------"""




# model = model.fit(training_data, validation_data, n_classes=6)
