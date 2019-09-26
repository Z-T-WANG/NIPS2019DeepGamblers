from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math
import numpy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, closefig
import dataset_utils

model_names = ("vgg16","vgg16_bn")

parser = argparse.ArgumentParser(description='PyTorch cifar10 / svhn / catsdogs Gambler training')
parser.add_argument('-d', '--dataset', default='cifar10', type=str, choices=['cifar10', 'svhn', 'catsdogs'])
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Training
parser.add_argument('-t', '--train', dest='evaluate', action='store_true',
                    help='train the model. When evaluate is true, training is ignored and trained models are loaded.')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[25,50,75,100,125,150,175,200,225,250,275],
                        help='Multiply learning rate by gamma at the scheduled epochs (default: 25,50,75,100,125,150,175,200,225,250,275)')
parser.add_argument('--gamma', type=float, default=0.5, help='LR is multiplied by gamma on schedule (default: 0.5)') 
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-o', '--rewards', dest='rewards', type=float, nargs='+', default=[2.2],
                    metavar='o', help='The reward o for a correct prediction; Abstention has a reward of 1. Provided parameters would be stored as a list for multiple runs.')
parser.add_argument('--pretrain', type=int, default=0,
                    help='Number of pretraining epochs using the cross entropy loss, so that the learning can always start. Note that it defaults to 100 if dataset==cifar10 and reward<6.1, and the results in the paper are reproduced.')
parser.add_argument('--coverage', type=float, nargs='+',default=[100.,99.,98.,97.,95.,90.,85.,80.,75.,70.,60.,50.,40.,30.,20.,10.],
                    help='the expected coverages used to evaluated the accuracies after abstention')                    
# Save
parser.add_argument('-s', '--save', default='save', type=str, metavar='PATH',
                    help='path to save checkpoint (default: save)')
#parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                    help='path to load the saved model (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_bn',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vgg16_bn) Please edit the code to train with other architectures')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_false',
                    help='evaluate trained models on validation set, following the paths defined by "save", "arch" and "rewards"')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# set the abstention definitions
expected_coverage = args.coverage
reward_list = args.rewards

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

num_classes=10 # this is modified later in main() when defining the specific datasets
def main():
    # make path for the current archtecture & reward
    if not resume_path and not os.path.isdir(save_path):
        mkdir_p(save_path)

    # Dataset
    print('==> Preparing dataset %s' % args.dataset)
    global num_classes
    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10
        num_classes = 10
        input_size = 32
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        trainset = dataset(root='./data', train=True, download=True, transform=transform_train)
        testset = dataset(root='./data', train=False, download=True, transform=transform_test)
    elif args.dataset == 'svhn':
        dataset = datasets.SVHN
        num_classes = 10
        input_size = 32
        transform_train = transforms.Compose([
                transforms.RandomCrop(32,padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        trainset = dataset(root='./data/svhn', train=True, download=True, transform=transform_train)
        testset = dataset(root='./data/svhn', train=False, download=True, transform=transform_test)
    elif args.dataset == 'catsdogs':
        num_classes = 2
        input_size = 64
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=6),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        # use the "ImageFolder" datasets
        assert os.path.exists("./data/cats_dogs/train") and os.path.exists("./data/cats_dogs/test"), "Please download and put the 'cats vs dogs' dataset to paths 'data/cats_dogs/train' and 'cats_dogs/test'"
        trainset =  datasets.ImageFolder('./data/cats_dogs/train')
        testset =  datasets.ImageFolder('./data/cats_dogs/test')
        # resizing the images to 64 and center crop them, so that they become 64x64 squares
        trainset = dataset_utils.resized_dataset(trainset, transform_train, resize=64)
        testset = dataset_utils.resized_dataset(testset, transform_test, resize=64)
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    # End of Dataset
    
    # Model
    print("==> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=num_classes+1, input_size = input_size)
   
    if use_cuda: model = torch.nn.DataParallel(model.cuda())
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if args.pretrain: criterion = nn.CrossEntropyLoss() 
    # the conventional loss is replaced by the gambler's loss in train() and test() explicitly except for pretraining
    optimizer = optim.SGD(model.parameters(), lr=state['lr'], momentum=args.momentum, weight_decay=args.weight_decay)


    title = args.dataset + '-' + args.arch + ' o={:.2f}'.format(reward)
    logger = Logger(os.path.join(save_path, 'log.txt'), title=title)
    logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Test Loss', 'Train Err.', 'Test Err.'])

    # if only for evaluation, the training part will not be executed
    if args.evaluate:
        print('\nEvaluation only')
        assert os.path.isfile(resume_path), 'no model exists at "{}"'.format(resume_path)
        model = torch.load(args.resume)
        if use_cuda: model = model.cuda()
        test(testloader, model, criterion, start_epoch, use_cuda, evaluation = True)
        return

    # train
    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('\n'+save_path)
        print('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        # save the model
        filepath = os.path.join(save_path, "{:d}".format(epoch+1) + ".pth")
        torch.save(model, filepath)
        # delete the last saved model if exist
        last_path = os.path.join(save_path, "{:d}".format(epoch) + ".pth")
        if os.path.isfile(last_path): os.remove(last_path)
        # append logger file
        logger.append([epoch+1, state['lr'], train_loss, test_loss, 100-train_acc, 100-test_acc])

    filepath = os.path.join(save_path, "{:d}".format(args.epochs) + ".pth")
    torch.save(model, filepath)
    last_path = os.path.join(save_path, "{:d}".format(args.epochs-1) + ".pth")
    if os.path.isfile(last_path): os.remove(last_path)
    logger.plot(['Train Loss', 'Test Loss'])
    savefig(os.path.join(save_path, 'logLoss.eps'))
    closefig()
    logger.plot(['Train Err.', 'Test Err.'])
    savefig(os.path.join(save_path, 'logErr.eps'))
    closefig()
    logger.close()

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx,  (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        if epoch >= args.pretrain:
            outputs = F.softmax(outputs, dim=1)
            outputs, reservation = outputs[:,:-1], outputs[:,-1]
            gain = torch.gather(outputs, dim=1, index=targets.unsqueeze(1)).squeeze()
            doubling_rate = (gain.add(reservation.div(reward))).log()

            loss = -doubling_rate.mean()
        else:
            loss = criterion(outputs[:,:-1], targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda, evaluation = False):
    global best_acc

    # whether to evaluate uncertainty, or confidence
    if evaluation:
        evaluate(testloader, model, use_cuda)
        return

    # switch to test mode
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        abstention_results = []
        with torch.no_grad():
            outputs = model(inputs)
            values, predictions = outputs.data.max(1)
            if epoch >= args.pretrain:
                outputs = F.softmax(outputs, dim=1)
                outputs, reservation = outputs[:,:-1], outputs[:,-1]
                # analyze the accuracy at different abstention level
                abstention_results.extend(zip(list( reservation.numpy() ),list( predictions.eq(targets.data).numpy() )))
                # calculate loss
                gain = torch.gather(outputs, dim=1, index=targets.unsqueeze(1)).squeeze()
                doubling_rate = (gain.add(reservation.div(reward))).log()
                loss = -doubling_rate.mean()
            else:
                loss = criterion(outputs[:,:-1], targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    if epoch >= args.pretrain:
    	# sort the abstention results according to their reservations, from high to low
    	abstention_results.sort(key = lambda x: x[0], reverse=True)
    	# get the "correct or not" list for the sorted results
    	sorted_correct = list(map(lambda x: int(x[1]), abstention_results))
    	size = len(testloader)
    	print('accracy of coverage ',end='')
    	for coverage in expected_coverage:
    	    print('{:.0f}: {:.3f}, '.format(coverage, sum(sorted_correct[int(size/100*coverage):])),end='')
    	print('')
    return (losses.avg, top1.avg)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
            
# this function is used to evaluate the accuracy on validation set and test set per coverage
def evaluate(testloader, model, use_cuda):
    model.eval()
    abortion_results = [[],[]]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets
            output = model(inputs)
            output = F.softmax(output,dim=1)
            output, reservation = output[:,:-1], (output[:,-1]).cpu()
            values, predictions = output.data.max(1)
            predictions = predictions.cpu()
            abortion_results[0].extend(list( reservation.numpy() ))
            abortion_results[1].extend(list( predictions.eq(targets.data).numpy() ))
    def shuffle_list(list, seed=10):
        random.seed(seed)
        random.shuffle(list)
    shuffle_list(abortion_results[0]); shuffle_list(abortion_results[1])
    abortion, correct = torch.tensor(abortion_results[0]), torch.tensor(abortion_results[1])
    # use 2000 data points as the validation set (randomly shuffled)
    abortion_valid, abortion = abortion[:2000], abortion[2000:]
    correct_valid, correct = correct[:2000], correct[2000:]
    results_valid=[]; results=[]
    bisection_method(abortion_valid, correct_valid, results_valid)
    bisection_method(abortion, correct, results)
    f = open(os.path.join(os.path.dirname(resume_path),'coverage VS err.csv'), 'w')
    for idx, result in enumerate(results):
        f.write('valid{:.0f},{:.2f},{:.3f}\n'.format(expected_coverage[idx],results_valid[idx][0]*100.,(1-results_valid[idx][1])*100))
        f.write('test{:.0f},{:.2f},{:.3f}\n'.format(expected_coverage[idx],results[idx][0]*100.,(1-results[idx][1])*100))
    return

def bisection_method(abortion, correct, results):
    upper = 1.
    while True:
        mask_up = abortion <=  upper
        passed_up = torch.sum(mask_up.long()).item()
        if passed_up/len(correct)*100.<expected_coverage[0]: upper *= 2.
        else: break
    test_thres = 1.
    for coverage in expected_coverage:
        mask = abortion <=  test_thres
        passed = torch.sum(mask.long()).item()
        # bisection method start
        lower = 0.
        while math.fabs(passed/len(correct)*100.-coverage) > 0.3:
            if passed/len(correct)*100.>coverage:
                upper = min(test_thres,upper)
                test_thres=(test_thres+lower)/2
            elif passed/len(correct)*100.<coverage:
                lower = max(test_thres,lower)
                test_thres=(test_thres+upper)/2
            mask = abortion <=  test_thres
            passed = torch.sum(mask.long()).item()
            # bisection method end
        masked_correct = correct[mask]
        correct_data = torch.sum(masked_correct.long()).item()
        passed_acc = correct_data/passed
        results.append((passed/len(correct), passed_acc))
        #print('coverage {:.0f} done'.format(coverage))

# this function is used to organize all data and write into one file
def save_data():
    save=open('{}_{}.csv'.format(args.dataset,args.arch),'w')
    save.write('0,100val.,100test,99v,99t,98v,98t,97v,97t,95v,95t,90v,90t,85v,85t,80v,80t,75v,75t,70v,70t,60v,60t,50v,50t,40v,40t,30v,30t,20v,20t,10v,10t\n')
    for reward in reward_list:
        save_path = base_path+'o={:.2f}'.format(reward)
        if os.path.isfile(os.path.join(save_path, 'coverage VS err.csv')):
            f = open(os.path.join(save_path, 'coverage VS err.csv') ,'r')
        else: print('no file exists at {}'.format(os.path.join(save_path, 'coverage VS err.csv'))); continue
        content = f.read()
        lines = content.split('\n')
        save.write('o={:.2f},'.format(reward))
        for idx in range(len(expected_coverage)):
            save.write('{},'.format(lines[2*idx].split(',')[2]))
            save.write('{},'.format(lines[2*idx+1].split(',')[2]))
        save.write('\n')
        f.close()

if __name__ == '__main__':
    base_path = os.path.join(args.save, args.dataset, args.arch)+'_gambling_'
    baseLR = state['lr']
    base_pretrain = args.pretrain
    resume_path = ""
    for i in range(len(reward_list)): 
        state['lr'] = baseLR
        reward = reward_list[i]
        save_path = base_path+'o={:.2f}'.format(reward)
        if args.evaluate: resume_path= os.path.join(save_path,'{:d}.pth'.format(args.epochs))
        args.pretrain = base_pretrain
        
        # default the pretraining epochs to 100 to reproduce the results in the paper
        if args.pretrain==0 and reward <6.1 and args.dataset=='cifar10': args.pretrain=100
        
        main()
    if args.evaluate: save_data()
    

