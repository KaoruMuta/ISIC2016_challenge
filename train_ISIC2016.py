#!/bin/python
from optparse import OptionParser

import os
import sys
import random
import shutil
import numpy as np
import itertools
from PIL import Image
import pandas as pd

import torch
from torch.autograd import Variable
from random import shuffle

import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from enum import Enum
# Custom imports
import pretrainedmodels
from cutout import *
from focalloss import *
import cv2
#from se_resnext import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Data(Enum):
    TRAIN = 1
    TEST = 2
    VALIDATION = 3

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, imagenumber=None, split=Data.TRAIN, transform=None):
        """
        Args:
            root_dir (string): Directory with all the pickle files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files = []
        self.trainfiles = []
        self.testfiles = []
        self.valfiles = []
        self.imageFormats = [".jpg", ".png", ".bmp", 'jpeg']
        self.split = split
        self.isic2016train = pd.read_csv('ISBI2016_ISIC_Part3_Training_GroundTruth.csv', header=None)
        self.isic2016test = pd.read_csv('ISBI2016_ISIC_Part3_Test_GroundTruth.csv', header=None)
        self.imagenumber = imagenumber
        self.count = 0

        #dataloading
        for root, dirs, files in os.walk('2016train'):
            for file in files:
                for imageFormat in self.imageFormats:
                    if file.endswith(imageFormat):
                        self.files.append(os.path.abspath(os.path.join(root, file)))
                        break

        for root, dirs, files in os.walk('2016test'):
            for file in files:
                for imageFormat in self.imageFormats:
                    if file.endswith(imageFormat):
                        self.testfiles.append(os.path.abspath(os.path.join(root, file)))
                        break

        random.seed(0)
        shuffle(self.files)
        shuffle(self.testfiles)

        for i in range(self.imagenumber):
            self.trainfiles.append(self.files[i])
        for i in range(self.imagenumber, 900):
            self.valfiles.append(self.files[i])

        self.transform = transform
        self.oneHot = False # Torch can't deal with one-hot vectors

    def getNumClasses(self):
        return 2

    def __len__(self): #the number of all files
        if self.split == Data.TRAIN:
            return len(self.trainfiles)

        elif self.split == Data.TEST:
            return len(self.testfiles)

        elif self.split == Data.VALIDATION:
            return len(self.valfiles)

    def __getitem__(self, idx):
        if self.split == Data.TRAIN:
            file = self.trainfiles[idx]
            img = Image.open(file)
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)

            for i in range(900):
                if file.split(os.sep)[-1] == self.isic2016train.iloc[i][0] + '.jpg':
                    if self.isic2016train.iloc[i][1] == 'benign':
                        label = 0
                    else:
                        label = 1
                    break

            if self.oneHot:
                oneHot = np.zeros(2)
                oneHot[label] = 1.0
                label = oneHot

            sample = {'data': img, 'label': label}

        elif self.split == Data.TEST:
            file = self.testfiles[idx]
            img = Image.open(file)
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)

            for i in range(379):
                if file.split(os.sep)[-1] == self.isic2016test.iloc[i][0] + '.jpg':
                    label = int(self.isic2016test.iloc[i][1])
                    break

            if self.oneHot:
                oneHot = np.zeros(self.getNumClasses())
                oneHot[label] = 1.0
                label = oneHot

            sample = {'data': img, 'label': label}

        elif self.split == Data.VALIDATION:
            file = self.valfiles[idx]
            img = Image.open(file)
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)

            for i in range(900):
                if file.split(os.sep)[-1] == self.isic2016train.iloc[i][0] + '.jpg':
                    if self.isic2016train.iloc[i][1] == 'benign':
                        label = 0
                    else:
                        label = 1
                    break

            if self.oneHot:
                oneHot = np.zeros(2)
                oneHot[label] = 1.0
                label = oneHot

            sample = {'data': img, 'label': label}

        return sample

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    print('bz', bz)
    print('nc', nc)
    print('h', h)
    print('w', w)

    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def train(options):
    # Clear output directory
    if os.path.exists(options.outputDir):
        print ("Removing old directory!")
        shutil.rmtree(options.outputDir)
    os.mkdir(options.outputDir)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()
    predictedLabels = []
    gtLabels = []
    plbs = []
    glbs = []
    # Create model
    if options.useTorchVisionModels:
        print('model')
        model = models.densenet161(pretrained=True)

        # Identify the name of the last layer
        for name, child in model.named_children():
            for name2, params in child.named_parameters():
                print(name, name2)

        finalconv_name = 'features'
        ## Change the last layer
        inputDim = model.classifier.in_features
        model.classifier = torch.nn.Linear(inputDim, options.numClasses)
        # TODO: Rectify the transform params
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_size = 224

    else:
        # Use pretrained models library -  (pip install --upgrade pretrainedmodels)
        # https://github.com/Cadene/pretrained-models.pytorch
        model = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')

        # Change the last layer
        inputDim = model.last_linear.in_features
        model.last_linear = torch.nn.Linear(inputDim, options.numClasses)

        mean = model.mean
        std = model.std
        input_size = model.input_size[1]

        finalconv_name = 'layer4'
        assert model.input_size[1] == model.input_size[2], "Error: Models expects different dimensions for height and width"
        assert model.input_space == "RGB", "Error: Data loaded in RGB format while the model expects BGR"

    if options.ft:
        model = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained=None)

        # Change the last layer
        inputDim = model.last_linear.in_features
        model.last_linear = torch.nn.Linear(inputDim, 642)

        print('fine tuning now')
        modelCheckpoint = torch.load(os.path.join('best_model', 'model_f5_SEResNeXt101.pth'))
        model.load_state_dict(modelCheckpoint)

        inputDim = model.last_linear.in_features
        model.last_linear = torch.nn.Linear(inputDim, options.numClasses)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_size = 224

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # Move the model to desired device
    model.to(device)

    # Create dataloader
    dataTransform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    if options.cutout:
        dataTransform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            Cutout(n_holes=1, length=16)]) #These are default values

    dataTransformVal = transforms.Compose([
        transforms.Resize(input_size),
        transforms.FiveCrop(input_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=mean, std=std)(crop) for crop in crops]))])

    print('loading train dataset')
    dataset = MyDataset(split=Data.TRAIN, imagenumber=options.imagenumber, transform=dataTransform)
    dataLoader = DataLoader(dataset=dataset, num_workers=0, batch_size=options.batchSize, shuffle=True)
    assert options.numClasses == dataset.getNumClasses(), "Error: Number of classes found in the dataset is not equal to the number of classes specified in the options (%d != %d)!" % (dataset.getNumClasses(), options.numClasses)

    print('loading validation dataset')
    datasetval = MyDataset(split=Data.VALIDATION, imagenumber=options.imagenumber, transform=dataTransform)
    dataLoaderval = DataLoader(dataset=datasetval, num_workers=0, batch_size=options.batchSize, shuffle=True)
    assert options.numClasses == datasetval.getNumClasses(), "Error: Number of classes found in the dataset is not equal to the number of classes specified in the options (%d != %d)!" % (datasetval.getNumClasses(), options.numClasses)

    print('loading test dataset')
    datasetVal = MyDataset(split=Data.TEST, imagenumber=options.imagenumber, transform=dataTransformVal)
    dataLoaderVal = DataLoader(dataset=datasetVal, num_workers=0, batch_size=options.batchSize, shuffle=False)

    assert options.numClasses == datasetVal.getNumClasses(), "Error: Number of classes found in the dataset is not equal to the number of classes specified in the options (%d != %d)!" % (datasetVal.getNumClasses(), options.numClasses)
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(options.trainingEpochs):
        model.train()
        train_loss = 0.0
        print('epoch start')
        # Start training
        for iterationIdx, data in enumerate(dataLoader):
            X = data["data"]
            y = data["label"]
            # Move the data to PyTorch on the desired device
            X = Variable(X).float().to(device)
            y = Variable(y).long().to(device)
            # Get model predictions
            pred = model(X)
            # Optimize
            optimizer.zero_grad()
            if options.focal == True:
                loss = FocalLoss(gamma=0.5)(pred, y)
            else:
                loss = criterion(pred, y)
            train_loss += loss.item()
            _, preds = torch.max(pred.data, dim = 1)

            plbs.append(preds.cpu().numpy())
            glbs.append(y.data.cpu().numpy())

            if plbs[int(iterationIdx/16)][iterationIdx] != glbs[int(iterationIdx/16)][iterationIdx]
                loss.backward()
                optimizer.step()

            if iterationIdx % options.displayStep == 0:
                print("Epoch %d | Iteration: %d | Loss: %.5f" % (epoch, iterationIdx, loss))

        print('train_Loss:', train_loss / len(plbs))
        with open(os.path.join(options.outputDir, 'log.txt'), 'a') as f:
            print(train_loss / len(plbs), file=f)

        for i in range(len(glbs)):
            for j in range(len(glbs[i])):
                gtLabels.append(glbs[i][j])

        for i in range(len(plbs)):
            for j in range(len(plbs[i])):
                predictedLabels.append(plbs[i][j])

        epoch_acc = accuracy_score(gtLabels, predictedLabels)
        print('train_acc:', epoch_acc)
        with open(os.path.join(options.outputDir, 'train_acc.txt'), 'a') as f5:
            print(epoch_acc, file=f5)

        scheduler.step()
        predictedLabels.clear()
        gtLabels.clear()
        plbs.clear()
        glbs.clear()
        # Save model
        if epoch == 1:
            torch.save(model.state_dict(), os.path.join(options.outputDir, "model_epoch2.pth"))
        if epoch == 3:
            torch.save(model.state_dict(), os.path.join(options.outputDir, "model_epoch4.pth"))
        if epoch == 5:
            torch.save(model.state_dict(), os.path.join(options.outputDir, "model_epoch6.pth"))
        if epoch == 7:
            torch.save(model.state_dict(), os.path.join(options.outputDir, "model_epoch8.pth"))
        if epoch == 9:
            torch.save(model.state_dict(), os.path.join(options.outputDir, "model_epoch10.pth"))

        model.eval()
        val_loss = 0
        for iterationIdx, data in enumerate(dataLoaderval):
            X = data["data"]
            y = data["label"]
            # Move the data to PyTorch on the desired device
            X = Variable(X).float().to(device)
            y = Variable(y).long().to(device)
            #testing the dataset
            bs, ncrops, c, h, w = X.size()
            with torch.no_grad():
                temp_output = model(X.view(-1, c, h, w))
            outputs = temp_output.view(bs, ncrops, -1).mean(1)

            _, preds = torch.max(outputs.data, dim = 1)
            if options.focal == True:
                loss = FocalLoss(gamma=0.5)(outputs, y)
            else:
                loss = criterion(outputs, y)
            val_loss += loss.item()
            plbs.append(preds.cpu().numpy())
            glbs.append(y.data.cpu().numpy())

        print('val_Loss:', val_loss / len(plbs))
        for i in range(len(glbs)):
            for j in range(len(glbs[i])):
                gtLabels.append(glbs[i][j])

        for i in range(len(plbs)):
            for j in range(len(plbs[i])):
                predictedLabels.append(plbs[i][j])

        epoch_acc = accuracy_score(gtLabels, predictedLabels)
        print('val_acc:', epoch_acc)

    correctExamples = 0
    oneHot = []
    pred_fold = []
    model.eval()
    #hooked feature map
    model._modules.get(finalconv_name).register_forward_hook(hook_feature)
    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
    for iterationIdx, data in enumerate(dataLoaderVal):
        X = data["data"]
        y = data["label"]
        # Move the data to PyTorch on the desired device
        X = Variable(X).float().to(device)
        y = Variable(y).long().to(device)
        #testing the dataset
        bs, ncrops, c, h, w = X.size()
        with torch.no_grad():
            temp_output = model(X.view(-1, c, h, w))
        outputs = temp_output.view(bs, ncrops, -1).mean(1)

        _, preds = torch.max(outputs.data, dim = 1)
        correctExamples += (preds == y.data).sum().item()
        #converting tensor to numpy
        plbs.append(preds.cpu().numpy())
        glbs.append(y.data.cpu().numpy())
        m = torch.nn.Softmax()
        outputs = m(outputs)
        #print('Softmax', outputs)
        outputs = outputs.cpu().numpy()
        for i in range(len(outputs)):
            pred_fold.append(outputs[i])

    for i in range(len(plbs)):
        for j in range(len(plbs[i])):
            predictedLabels.append(plbs[i][j])
            gtLabels.append(glbs[i][j])

    for i in range(len(gtLabels)):
        oh = np.zeros(2)
        oh[gtLabels[i]] = 1.0
        oneHot.append(oh)

    oneHot = np.asarray(oneHot)
    pred_fold = np.asarray(pred_fold)

    Classes=[0,1]
    accuracy = accuracy_score(gtLabels, predictedLabels)
    print("Correct examples: %d | Total examples: %d | Accuracy: %.5f" % (correctExamples, len(predictedLabels), float(correctExamples) / len(predictedLabels)))
    print(classification_report(gtLabels, predictedLabels))
    print('accuracy:', accuracy)
    cnf_matrix = confusion_matrix(gtLabels, predictedLabels)
    plot_cfmatrix(cnf_matrix, classes=Classes, title='Confusion matrix', cmap=plt.cm.RdPu)

    fpr, tpr, thresholds = roc_curve(oneHot[:, 1], pred_fold[:, 1])
    roc_auc = auc(fpr, tpr)
    print('AUC of fold:', roc_auc)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(options.outputDir, 'roc_curve.png'))

    with open(os.path.join(options.outputDir, 'accuracy.txt'), 'a') as f3:
        print(accuracy, file=f3)
        print(roc_auc, file=f3)
        print(classification_report(gtLabels, predictedLabels), file=f3)

    with open(os.path.join(options.outputDir, 'gtLabel.txt'), 'a') as gtchecking:
        for idx in range(len(gtLabels)):
            print(gtLabels[idx], file=gtchecking)
    with open(os.path.join(options.outputDir, 'predictedLabel.txt'), 'a') as predchecking:
        for idx in range(len(predictedLabels)):
            print(predictedLabels[idx], file=predchecking)

    # generate class activation mapping for the top1 prediction
    '''CAMs = returnCAM(features_blobs[0], weight_softmax, predictedLabels)

    # render the CAM and output
    img = cv2.imread('2016test/ISIC_0000003.jpg')
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('CAM.jpg', result)'''

    predictedLabels.clear()
    gtLabels.clear()
    plbs.clear()
    glbs.clear()

def plot(options):
    tloss_x = []
    tloss_y = []
    t_accx = []
    t_accy = []
    with open(os.path.join(options.outputDir, 'log.txt')) as f1:
        for i, line in enumerate(f1):
            tloss_x.append(i)
            tloss_y.append(float(line))

    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    #plt.xlim(0, options.trainingEpochs+0.5)
    plt.plot(tloss_x, tloss_y, label='train_loss')
    #plt.plot(val_x, val_y, label='val_loss')
    xlabels = [0, int(options.trainingEpochs / 2), options.trainingEpochs]
    plt.xticks(xlabels, xlabels)
    plt.legend()
    plt.savefig(os.path.join(options.outputDir, 'loss.png'))

    with open(os.path.join(options.outputDir, 'train_acc.txt')) as f3:
        for i, line in enumerate(f3):
            t_accx.append(i)
            t_accy.append(float(line))

    plt.figure()
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    #plt.xlim(0, options.trainingEpochs+0.5)
    #plt.ylim(0, 1)
    plt.plot(t_accx, t_accy, label='train_acc')
    #plt.plot(val_accx, val_accy, label='val_acc')
    xlabels = [0, int(options.trainingEpochs / 2), options.trainingEpochs]
    plt.xticks(xlabels, xlabels)
    plt.legend()
    plt.savefig(os.path.join(options.outputDir, 'accuracy.png'))

def plot_cfmatrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.RdPu):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[np:newaxis]
        print('Normlized confusion matrix')
    else:
        print('confusion matrix')

    print(cm)

    plt.figure(figsize=(12,12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.savefig(os.path.join(options.outputDir, 'confusion_matrix.png'))


if __name__ == "__main__":
    # Command line options
    parser = OptionParser()

    # Base options
    #parser.add_option("-m", "--model", action="store", type="string", dest="model", default="NAS", help="Model to be used for Cross-Layer Pooling")
    parser.add_option("-t", "--trainModel", action="store_true", dest="trainModel", default=False, help="Train model")
    #parser.add_option("-c", "--testModel", action="store_true", dest="testModel", default=False, help="Test model")
    parser.add_option("-o", "--outputDir", action="store", type="string", dest="outputDir", default="./output", help="Output directory")
    parser.add_option("-e", "--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=10, help="Number of training epochs")
    parser.add_option("-b", "--batchSize", action="store", type="int", dest="batchSize", default=32, help="Batch Size")
    parser.add_option("-d", "--displayStep", action="store", type="int", dest="displayStep", default=2, help="Display step where the loss should be displayed")
    parser.add_option('-p', '--plot', action='store_true', dest='plot', default=False, help='plot')
    parser.add_option('-i', '--imagenumber', action='store', type='int', dest='imagenumber', default=900, help='imagenumber')

    # Input Reader Params
    #parser.add_option("--rootDir", action="store", type="string", dest="rootDir", default="../data/", help="Root directory containing the data")
    parser.add_option("--numClasses", action="store", type="int", dest="numClasses", default=2, help="Number of classes in the dataset")
    parser.add_option("--useTorchVisionModels", action="store_true", dest="useTorchVisionModels", default=False, help="Use pre-trained models from the torchvision library")
    parser.add_option("--ft", action="store_true", dest="ft", default=False, help="Use pre-trained models from DermNet")
    parser.add_option("--cutout", action="store_true", dest="cutout", default=False, help="applying cutout")
    parser.add_option("--focal", action="store_true", dest="focal", default=False, help="applying focal loss")

    #parser.add_option('--lr', action='store', type='float', dest='learning rate', default=0.0001, help='learning rate')

    # Parse command line options
    (options, args) = parser.parse_args()
    print(options)

    if options.trainModel:
        print("Training model")
        train(options)

    if options.plot:
        print('plot now')
        plot(options)
