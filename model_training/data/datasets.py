import numpy as np 
import torchvision.transforms as transforms
from .cifar import CIFAR10, CIFAR100,CIFAR10N_resplit
from .face_attr import face_attributes



train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_cifar100_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

train_face_attr_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4801, 0.4250, 0.4140], std=[0.3512, 0.3312, 0.3284])
    ])

test_face_attr_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4801, 0.4250, 0.4140], std=[0.3512, 0.3312, 0.3284])
    ])

def input_dataset(dataset, noise_type, noise_path, is_human,root=None,face_attr_class_type=None):
    if dataset == 'cifar10':
        train_dataset = CIFAR10(root='~/data/',
                                download=True,  
                                train=True, 
                                transform = train_cifar10_transform,
                                noise_type = noise_type,
                                noise_path = noise_path, is_human=is_human
                           )
        test_dataset = CIFAR10(root='~/data/',
                                download=False,  
                                train=False, 
                                transform = test_cifar10_transform,
                                noise_type=noise_type
                          )
        num_classes = 10
        num_training_samples = 50000
        
    elif dataset == 'cifar10_nr':
        train_dataset = CIFAR10N_resplit(root='~/data/',
                                download=True,  
                                train=True, 
                                transform = train_cifar10_transform,
                                noise_type = noise_type,
                                noise_path = noise_path, is_human=is_human,
                                remix=False
                        )
        a = train_dataset.select_dict
        test_dataset = CIFAR10N_resplit(root='~/data/',
                                download=False,  
                                train=False, 
                                transform = test_cifar10_transform,
                                noise_type=noise_type,
                                noise_path = noise_path, is_human=is_human,
                                select_dict=train_dataset.select_dict,
                                remix=False
                        )
        num_classes = 10
        num_training_samples = 40000 #16350#40000
    elif dataset == 'cifar100':
        train_dataset = CIFAR100(root='~/data/',
                                download=True,  
                                train=True, 
                                transform=train_cifar100_transform,
                                noise_type=noise_type,
                                noise_path = noise_path, is_human=is_human
                            )
        test_dataset = CIFAR100(root='~/data/',
                                download=False,  
                                train=False, 
                                transform=test_cifar100_transform,
                                noise_type=noise_type
                            )
        num_classes = 100
        num_training_samples = 50000
    return train_dataset, test_dataset, num_classes, num_training_samples


def input_dataset_face_attr(args, dataset,root=None,human_dir=None):
    if 'resnet' in args.model :
        face_attr_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif 'vit' in args.model:
        face_attr_transform = transforms.Compose([
            transforms.Resize((384, 384)), 
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    if dataset =='face_attribute':
        # train_dataset = face_attributes(root=root,
        #                         train=True,
        #                         transform=train_face_attr_transform,
        #                         class_type=face_attr_class_type+'_train',
        #                         debug=False)
        # test_dataset = face_attributes(root=root,
        #                 train=False,
        #                 transform=test_face_attr_transform,
        #                 class_type=face_attr_class_type+'_test',
        #                 debug=False)

        # TODO fix this hard code path
        # root = '/home/mikelmh025/Documents/data/navi_data/'
        # human_dir = 'FairFace2.0/'
        root = args.data_root
        human_dir = args.human_dataset
        debug = args.debug
        train_dataset = face_attributes(root,human_dir,debug=debug,train_mode='train',target_mode=args.target_mode,transform=face_attr_transform)
        test_dataset = face_attributes(root,human_dir,debug=debug,train_mode='val',target_mode=args.target_mode,transform=face_attr_transform)

        num_classes = train_dataset.num_classes
        num_training_samples = train_dataset.__len__()

    return train_dataset, test_dataset, num_classes, num_training_samples


def input_dataset_face_attr_test(args, dataset,root=None,human_dir=None):
    if 'resnet' in args.model :
        face_attr_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif 'vit' in args.model:
        face_attr_transform = transforms.Compose([
            transforms.Resize((384, 384)), 
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])
        
    if dataset =='face_attribute':
        root = args.data_root
        human_dir = args.human_dataset
        debug = args.debug
        test_dataset = face_attributes(root,human_dir,debug=debug,train_mode='test',target_mode=args.target_mode,transform=face_attr_transform)

        num_classes = test_dataset.num_classes
        num_training_samples = test_dataset.__len__()

    return test_dataset, num_classes, num_training_samples







