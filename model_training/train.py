

# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from data.datasets import input_dataset, input_dataset_face_attr
from models import *
import argparse
import numpy as np


from loss import forward_loss, loss_cross_entropy,loss_spl, f_beta, f_alpha_hard, lq_loss, loss_peer, loss_cores, forward_loss,mse_loss, softmax_cross_entropy_with_softtarget
from torch.utils.data import RandomSampler
from variable_optim import VSGD

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.05)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--loss', type = str, help = 'ce, gce, dmi, flc, uspl,spl,peerloss', default = 'ce')
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = '/home/jovyan/results')
# parser.add_argument('--noise_type', type = str, help='clean_label, aggre_label, worse_label, random_label1, random_label2, random_label3', default='clean_label')
# parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default='./data/noise_label/CIFAR-10_human.pt')
# parser.add_argument('--top_bn', action='store_true')
# parser.add_argument('--ideal', action='store_true')
parser.add_argument('--dataset', type = str, help = ' cifar10 or fakenews', default = 'cifar10')
parser.add_argument('--model', type = str, help = 'cnn,resnet,vgg', default = 'resnet')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=0, help='how many subprocesses to use for data loading')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# parser.add_argument('--is_human', action='store_true', default=False)
parser.add_argument('--device', type=str, help='cuda or cpu ', default='cuda')
parser.add_argument('--data_root', type=str, help='path to dataset', default='/media/otter/navi_hitl/')
parser.add_argument('--human_dataset', type=str, help='path to dataset', default='FairFace2.0/')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
# parser.add_argument('--model_path', type = str, default = 'cifar_ce.pt')

# Golbal variables of attributes
continuous_attr_dict = {
    'top_curly':4,
    'top_length':6,
    'side_curly':4,
    'side_length':5
}
discrete_attr_dict = {
    'braid_tf':2,
    'braid_type':5,
    'braid_count':4,
    'braid_position':3
}
multi_class_attr_dict = {
    'top_direction':8
}

# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch,alpha_plan,loss_type='cores'):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]/(1+f_beta(epoch,loss_type))
        

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_labels(pred,label,label_index_dict):
    # pred = pred.clone().detach().cpu().numpy()
    # label = label.clone().detach().cpu().numpy()

    # Intial total loss tensor
    total_loss = []
    batch_size = pred.shape[0]

    correct = {}

    for label_type in label_index_dict:
        pred_cur = pred[:,label_index_dict[label_type][0]:label_index_dict[label_type][1]+1]
        label_cur = label[:,label_index_dict[label_type][0]:label_index_dict[label_type][1]+1]
        if label_type in continuous_attr_dict:
            total_loss += [mse_loss(pred_cur,label_cur)] 
            diff = torch.abs(pred_cur-label_cur)
            correct[label_type] = torch.sum(diff<=0.5).item()
        elif label_type in discrete_attr_dict:
            total_loss += [loss_cross_entropy(pred_cur,label_cur)]
            pred_max = torch.argmax(pred_cur,dim=1)
            label_max = torch.argmax(label_cur,dim=1)
            correct[label_type] = torch.sum(pred_max==label_max).item()
        elif label_type in multi_class_attr_dict:
            # TODO: Fix this
            continue
            total_loss += [loss_cross_entropy(pred_cur,label_cur)]
            pred_max = torch.argmax(pred_cur,dim=1)
            label_max = torch.argmax(label_cur,dim=1)
            correct[label_type] = torch.sum(pred_max==label_max).item()

    total_prec = sum(correct.values())/(batch_size*len(correct))
    total_loss = sum(total_loss)
    return total_loss, correct, total_prec

# Train the Model
def train(epoch, train_loader, model, optimizer, train_dataset,label_index_dict):
    train_total=0
    train_correct=0

    batch_size = train_loader.batch_size
    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        

        images = Variable(images).to(args.device)
        labels = Variable(labels).to(args.device)
       
        # Forward + Backward + Optimize
        logits = model(images)

        loss,correct_dict, prec = accuracy_labels(logits,labels,label_index_dict)
        # prec, _ = accuracy(logits, labels, topk=(1, 5))

        train_total+=1
        train_correct+=prec

        # loss = mse_loss(logits,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data))
            report = ''
            for label_type in correct_dict:
                report += label_type + ": "+str(100*correct_dict[label_type]/batch_size) + ' || '
            print('report(Last batch)',report)


    train_acc=100*float(train_correct)/float(train_total)
    return train_acc, loss

# Evaluate the Model
def evaluate(test_loader,model,save=False,epoch=0,best_acc_=0,args=None,save_dir=None,label_index_dict=None):
    model.eval()    # Change model to 'eval' mode.
    print('Start evluate previous_best', best_acc_)
    correct = 0
    total = 0
    batch_size = test_loader.batch_size
    print_report = {}
    for images, labels, _ in test_loader:
        images = Variable(images).to(args.device)
        labels = Variable(labels).to(args.device)
        logits = model(images)
        # outputs = F.softmax(logits, dim=1)
        # _, pred = torch.max(outputs.data, 1)
        loss,correct_dict, prec = accuracy_labels(logits,labels,label_index_dict)
        for key in correct_dict:
            if key not in print_report: print_report[key] = []
            print_report[key].append(correct_dict[key])

        total += labels.size(0)
        correct += prec*labels.size(0) #(logits.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)
    print('Epoch [%d/%d] Test Accuracy: %.4f' % (epoch+1, args.n_epoch, acc))
    report = ''
    for label_type in print_report:
        report += label_type + ": "+str(100*sum(print_report[label_type])/total) + ' || '
    print('Test report(Last batch)',report)

    if save:
        if acc > best_acc_:
            state = {'state_dict': model.state_dict(),
                     'epoch':epoch,
                     'acc':acc,
            }
            # save_path= os.path.join(save_dir,args.loss + args.noise_type +'best.pth.tar')
            save_path= os.path.join(save_dir,args.loss +'best.pth.tar')
            torch.save(state,save_path)
            best_acc_ = acc
            print(f'model saved to {save_path}!')

            state = model.state_dict()
            save_path= os.path.join(save_dir,args.loss +'best.pt')
            torch.save(state,save_path)
        if epoch == args.n_epoch -1:
            state = {'state_dict': model.state_dict(),
                     'epoch':epoch,
                     'acc':acc,
            }
            torch.save(state,os.path.join(save_dir,args.loss +'last.pth.tar'))
    return acc, best_acc_

def main(args):

    # with wandb.init(config=config):
    #     config = wandb.config
    #     a=1
    # print(args.is_human)

    batch_size = args.batch_size
    learning_rate = args.lr
    # noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
    # args.noise_type = noise_type_map[args.noise_type]
    # load dataset
    train_dataset,test_dataset,num_classes,num_training_samples = input_dataset_face_attr(args, args.dataset,root=None,human_dir='v3')
    label_index_dict = train_dataset.get_index_dict()

    print('train_labels:', len(train_dataset))
    # load model
    print('building model...')
    if args.model == 'cnn':
        model = CNN(input_channel=3, n_outputs=num_classes)
    if args.model == 'vgg':
        model = vgg11()
    elif args.model == 'inception':
        model = Inception3()
    else:
        # model = ResNet34(num_classes)
        # model = models.resnet34(pretrained=True)
        model = resnet_pre(num_classes)
        # model = resnet_256(num_classes)

    # # load pretrained model
    # if args.pretrained:
    #     print('loading pretrained model...')
    #     model.load_state_dict(torch.load(args.pretrained))

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
    # optimizer = VSGD(model.parameters(), lr=learning_rate, variability=0.05, num_iters=math.ceil(50000/ batch_size) , weight_decay=1e-4)
    # optimizer = VSGD(model.parameters(), lr=0.1, num_iters=math.ceil(40000/ batch_size), variability = 0.04, weight_decay=1e-4)
    lambda_lr = lambda epoch: 0.1 ** (epoch // 100)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    # Creat loss and loss_div for each sample at each epoch
    loss_all = np.zeros((num_training_samples,args.n_epoch))
    loss_div_all = np.zeros((num_training_samples,args.n_epoch))
    ### save result and model checkpoint #######
    save_dir = args.result_dir +'/' +args.dataset + '/' + args.model
    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                    batch_size = batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size = batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False)
    
    # nn.DataParallel(model)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # model.to(device)
    model.to(args.device)
    # txtfile=save_dir + '/' +  args.loss + args.noise_type + '.txt'
    txtfile=save_dir + '/' +  args.loss + '.txt'
    if os.path.exists(txtfile):
        os.system('rm %s' % txtfile)

    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc test_acc \n')

    epoch=0
    train_acc = 0
    best_acc_ = 0.0
    #print(best_acc_)
    # training
    for epoch in range(args.n_epoch):
    # train models
        print(f'epoch {epoch}')
        # test_acc, best_acc_ = evaluate(test_loader=test_loader, save=True, model=model,epoch=epoch,best_acc_=best_acc_,args=args,save_dir=save_dir,label_index_dict=label_index_dict)
        # adjust_learning_rate(optimizer, epoch, alpha_plan, loss_type=args.loss)
        model.train()
        train_acc, train_loss = train(epoch, train_loader, model, optimizer, train_dataset,label_index_dict)
        scheduler.step()

    # evaluate models
        test_acc, best_acc_ = evaluate(test_loader=test_loader, save=True, model=model,epoch=epoch,best_acc_=best_acc_,args=args,save_dir=save_dir,label_index_dict=label_index_dict)

    # save results
        print('train acc on train images is ', train_acc, "loss is", train_loss)
        print('test acc on test images is ', test_acc)

        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + str(train_acc) +' ' + str(test_acc) + "\n")


# def set_sweep(args):
    



#     return sweep_id

# main function
if __name__ == '__main__':
    print('start')
    # import wandb
    # wandb.login()
    
    #####################################main code ################################################
    # args = parser.parse_args()

    # conver args to dict
    # args_dict = vars(args)

    # Seed
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(233) # This is the CIFAR-10N resplit seed
    # # Sweep config
    # sweep_config = {
    #     'method': 'random'
    #     }
    # metric = {
    #     'name': 'loss',
    #     'goal': 'minimize'   
    #     }
    # sweep_config['metric'] = metric
    # parameters_dict = {
    #     'test' :{
    #         'values': [args.loss]
    #     }
    # }
    # sweep_config['parameters'] = parameters_dict
    # parameters_dict.update({
    # 'learning_rate': {
    #     # a flat distribution between 0 and 0.1
    #     'distribution': 'uniform',
    #     'min': 0,
    #     'max': 0.1
    #   },
    # 'batch_size': {
    #     # integers between 32 and 256
    #     # with evenly-distributed logarithms 
    #     'distribution': 'q_log_uniform_values',
    #     'q': 8,
    #     'min': 4,
    #     'max': 32,
    #   }
    # })
    # import pprint
    # pprint.pprint(sweep_config)
    # sweep_id = wandb.sweep(sweep_config, project="hair-prediction")

    # wandb.agent(sweep_id, main, count=5)


    
    # start the main function
    main(args)
    