from optparse import OptionParser
import os, sys
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume
from core import model, dataset
from core.utils import init_log, progress_bar

sys.path.append(os.pardir)
import pretrainedmodels
#from focalloss import *

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# Command line options
parser = OptionParser()

# Base options
parser.add_option("-s", "--save_dir", action="store", type="string", dest="save_dir", default="./output", help="Output directory")
parser.add_option("-e", "--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=10, help="Number of training epochs")
parser.add_option("-b", "--batchSize", action="store", type="int", dest="batchSize", default=16, help="Batch Size")
#Input Reader Params
parser.add_option("--ft", action="store_true", dest="ft", default=False, help="Use pre-trained models from DermNet")
parser.add_option("--cutout", action="store_true", dest="cutout", default=False, help="applying cutout")
parser.add_option("--focal", action="store_true", dest="focal", default=False, help="applying focal loss")

# Parse command line options
(options, args) = parser.parse_args()
print(options)

start_epoch = 1
if os.path.exists(options.save_dir):
    raise NameError('model dir exists!')
os.makedirs(options.save_dir)
logging = init_log(options.save_dir)
_print = logging.info

# read dataset
trainset = dataset.CUB(root_dir='core/2016train', is_train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=options.batchSize,
                                          shuffle=True, num_workers=0, drop_last=False)
testset = dataset.CUB(root_dir='core/2016test', is_train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=options.batchSize,
                                         shuffle=False, num_workers=0, drop_last=False)
# define model
net = model.attention_net(topN=PROPOSAL_NUM)
if resume:
    ckpt = torch.load(resume)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1
creterion = torch.nn.CrossEntropyLoss()

# define optimizers
raw_parameters = list(net.pretrained_model.parameters())
part_parameters = list(net.proposal_net.parameters())
concat_parameters = list(net.concat_net.parameters())
partcls_parameters = list(net.partcls_net.parameters())

raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)
concat_optimizer = torch.optim.SGD(concat_parameters, lr=LR, momentum=0.9, weight_decay=WD)
part_optimizer = torch.optim.SGD(part_parameters, lr=LR, momentum=0.9, weight_decay=WD)
partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=LR, momentum=0.9, weight_decay=WD)
schedulers = [MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(concat_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(part_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(partcls_optimizer, milestones=[60, 100], gamma=0.1)]
net = net.cuda()
net = DataParallel(net)

for epoch in range(start_epoch, options.trainingEpochs+1):
    for scheduler in schedulers:
        scheduler.step()
    ##########################  train the model  ###############################
    _print('--' * 50)
    net.train()
    for i, data in enumerate(trainloader):
        img, label = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        raw_optimizer.zero_grad()
        part_optimizer.zero_grad()
        concat_optimizer.zero_grad()
        partcls_optimizer.zero_grad()

        raw_logits, concat_logits, part_logits, _, top_n_prob = net(img)
        part_loss = model.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                    label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size, PROPOSAL_NUM)
        if options.focal:
            raw_loss = FocalLoss(gamma=0.5)(raw_logits, label)
            concat_loss = FocalLoss(gamma=0.5)(concat_logits, label)
            rank_loss = model.ranking_loss(top_n_prob, part_loss)
            partcls_loss = FocalLoss(gamma=0.5)(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                     label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))
        else:
            raw_loss = creterion(raw_logits, label)
            concat_loss = creterion(concat_logits, label)
            rank_loss = model.ranking_loss(top_n_prob, part_loss)
            partcls_loss = creterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                     label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))

        total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
        total_loss.backward()
        raw_optimizer.step()
        part_optimizer.step()
        concat_optimizer.step()
        partcls_optimizer.step()
        progress_bar(i, len(trainloader), 'train')

    ##########################  evaluate net and save model  ###############################
    if epoch % SAVE_FREQ == 0:
        ##########################  evaluate net on test set  ###############################
        train_loss = 0
        train_correct = 0
        total = 0
        net.eval()
        for i, data in enumerate(trainloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                _, concat_logits, _, _, _ = net(img)
                # calculate loss
                if options.focal:
                    concat_loss = FocalLoss(gamma=0.5)(concat_logits, label)
                else:
                    concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                train_correct += torch.sum(concat_predict.data == label.data)
                train_loss += concat_loss.item() * batch_size
                progress_bar(i, len(trainloader), 'eval train set')

        train_acc = float(train_correct) / total
        train_loss = train_loss / total

        _print(
            'epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample: {}'.format(
                epoch,
                train_loss,
                train_acc,
                total))

        ##########################  evaluate net on test set  ###############################
        test_loss = 0
        test_correct = 0
        total = 0
        for i, data in enumerate(testloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                _, concat_logits, _, _, _ = net(img)
                # calculate loss
                if options.focal:
                    concat_loss = FocalLoss(gamma=0.5)(concat_logits, label)
                else:
                    concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                test_correct += torch.sum(concat_predict.data == label.data)
                test_loss += concat_loss.item() * batch_size
                progress_bar(i, len(testloader), 'eval test set')

        test_acc = float(test_correct) / total
        test_loss = test_loss / total
        _print(
            'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
                epoch,
                test_loss,
                test_acc,
                total))

        ##########################  save model  ###############################
        net_state_dict = net.module.state_dict()
        if not os.path.exists(options.save_dir):
            os.mkdir(options.save_dir)
        torch.save({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'net_state_dict': net_state_dict},
            os.path.join(options.save_dir, '%03d.ckpt' % epoch))

print('finishing training')
