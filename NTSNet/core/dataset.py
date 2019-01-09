import numpy as np
import scipy.misc
import os
from PIL import Image
from torchvision import transforms
import pandas as pd

class CUB():
    def __init__(self, is_train=True):
        self.train_img = []
        self.train_label = []
        self.test_img = []
        self.test_label = []
        self.imageFormats = [".jpg", ".png", ".bmp", 'jpeg']
        self.isic2016train = pd.read_csv('ISBI2016_ISIC_Part3_Training_GroundTruth.csv', header=None)
        self.isic2016test = pd.read_csv('ISBI2016_ISIC_Part3_Test_GroundTruth.csv', header=None)
        self.is_train = is_train
        if self.is_train:
            #dataloading
            for root, dirs, files in os.walk('2016train'):
                for file in files:
                    for imageFormat in self.imageFormats:
                        if file.endswith(imageFormat):
                            self.train_img.append(os.path.abspath(os.path.join(root, file)))
                            break

            print(len(self.train_img))
            for j in range(len(self.train_img)):
                for i in range(900):
                    if self.train_img[j].split(os.sep)[-1] == self.isic2016train.iloc[i][0] + '.jpg':
                        if self.isic2016train.iloc[i][1] == 'benign':
                            self.train_label.append(0)
                        else:
                            self.train_label.append(1)
            print(len(self.train_label))

        if not self.is_train:
            for root, dirs, files in os.walk('2016test'):
                for file in files:
                    for imageFormat in self.imageFormats:
                        if file.endswith(imageFormat):
                            self.test_img.append(os.path.abspath(os.path.join(root, file)))
                            break

            for j in range(len(self.test_img)):
                for i in range(379):
                    if self.test_img[j].split(os.sep)[-1] == self.isic2016test.iloc[i][0] + '.jpg':
                        self.test_label.append(int(self.isic2016test.iloc[i][1]))

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            '''if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)'''
            img = Image.open(img)
            img = img.convert('RGB')
            #img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomResizedCrop(224, scale=(0.7, 1.0))(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            #img = Cutout(n_holes=2, length=16)(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            '''if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')'''
            img = Image.open(img)
            img = img.convert('RGB')
            #img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.Resize(256)(img)
            img = transforms.CenterCrop(224)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


if __name__ == '__main__':
    dataset = CUB()
    print(len(dataset.train_img))
    print(len(dataset.train_label))
    for data in dataset:
        print(data[0].size(), data[1])
    dataset = CUB(is_train=False)
    print(len(dataset.test_img))
    print(len(dataset.test_label))
    for data in dataset:
        print(data[0].size(), data[1])
