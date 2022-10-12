from __future__ import print_function
from operator import delitem
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from .utils import download_url, check_integrity, noisify,noisify_instance, multiclass_noisify

class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_path = None, is_human=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset='cifar10'
        self.noise_type=noise_type
        self.nb_classes=10
        self.noise_path = noise_path
        idx_each_class_noisy = [[] for i in range(10)]
        if download:
           self.download()


        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            #if noise_type is not None:
            if noise_type !='clean':
                # Load human noisy labels
                train_noisy_labels = self.load_label()
                self.train_noisy_labels = train_noisy_labels.tolist()
                print(f'noisy labels loaded from {self.noise_path}')

                if not is_human:
                    T = np.zeros((self.nb_classes,self.nb_classes))
                    for i in range(len(self.train_noisy_labels)):
                        T[self.train_labels[i]][self.train_noisy_labels[i]] += 1
                    T = T/np.sum(T,axis=1)
                    print(f'Noise transition matrix is \n{T}')
                    train_noisy_labels = multiclass_noisify(y=np.array(self.train_labels), P=T,
                                        random_state=0) #np.random.randint(1,10086)
                    self.train_noisy_labels = train_noisy_labels.tolist()
                    T = np.zeros((self.nb_classes,self.nb_classes))
                    for i in range(len(self.train_noisy_labels)):
                        T[self.train_labels[i]][self.train_noisy_labels[i]] += 1
                    T = T/np.sum(T,axis=1)
                    print(f'New synthetic noise transition matrix is \n{T}')
    
                for i in range(len(self.train_noisy_labels)):
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
                self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
                print(f'The noisy data ratio in each class is {self.noise_prior}')
                self.noise_or_not = np.transpose(self.train_noisy_labels)!=np.transpose(self.train_labels)
                self.actual_noise_rate = np.sum(self.noise_or_not)/50000
                print('over all noise rate is ', self.actual_noise_rate)
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def load_label(self):
        #NOTE only load manual training label
        noise_label = torch.load(self.noise_path)
        if isinstance(noise_label, dict):
            if "clean_label" in noise_label.keys():
                clean_label = torch.tensor(noise_label['clean_label'])
                assert torch.sum(torch.tensor(self.train_labels) - clean_label) == 0  
                print(f'Loaded {self.noise_type} from {self.noise_path}.')
                print(f'The overall noise rate is {1-np.mean(clean_label.numpy() == noise_label[self.noise_type])}')
            return noise_label[self.noise_type].reshape(-1)  
        else:
            raise Exception('Input Error')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if self.noise_type !='clean':
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
 

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_rate=0.2, random_state=0,noise_path = None, is_human = True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset='cifar100'
        self.noise_type=noise_type
        self.nb_classes=100
        self.noise_path = noise_path
        idx_each_class_noisy = [[] for i in range(100)]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            if noise_type !='clean':
                # load noise label
                train_noisy_labels = self.load_label()
                self.train_noisy_labels = train_noisy_labels.tolist()
                print(f'noisy labels loaded from {self.noise_type}')
                if not is_human:
                    T = np.zeros((self.nb_classes,self.nb_classes))
                    for i in range(len(self.train_noisy_labels)):
                        T[self.train_labels[i]][self.train_noisy_labels[i]] += 1
                    T = T/np.sum(T,axis=1)
                    print(f'Noise transition matrix is \n{T}')
                    train_noisy_labels = multiclass_noisify(y=np.array(self.train_labels), P=T,
                                        random_state=0) #np.random.randint(1,10086)
                    self.train_noisy_labels = train_noisy_labels.tolist()
                    T = np.zeros((self.nb_classes,self.nb_classes))
                    for i in range(len(self.train_noisy_labels)):
                        T[self.train_labels[i]][self.train_noisy_labels[i]] += 1
                    T = T/np.sum(T,axis=1)
                    print(f'New synthetic noise transition matrix is \n{T}')
                for i in range(len(self.train_labels)):
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(100)]
                self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
                print(f'The noisy data ratio in each class is {self.noise_prior}')
                self.noise_or_not = np.transpose(self.train_noisy_labels)!=np.transpose(self.train_labels)
                self.actual_noise_rate = np.sum(self.noise_or_not)/50000
                print('over all noise rate is ', self.actual_noise_rate)
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC


import numpy as np
import random
class CIFAR10N_resplit(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'

    all_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_path = None, is_human=True,select_dict={},remix=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset='cifar10'
        self.noise_type=noise_type
        self.nb_classes=10
        self.noise_path = noise_path
        idx_each_class_noisy = [[] for i in range(10)]
        if download:
           self.download()

        self.remix=remix

        self.all_data = []
        self.all_labels = []
        for fentry in self.all_list:
            f = fentry[0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.all_data.append(entry['data'])
            if 'labels' in entry:
                self.all_labels += entry['labels']
            else:
                self.all_labels += entry['fine_labels']
            fo.close()

        self.all_data = np.concatenate(self.all_data)
        self.all_data = self.all_data.reshape((50000, 3, 32, 32))
        self.all_data = self.all_data.transpose((0, 2, 3, 1))  # convert to HWC
        #if noise_type is not None:
        if noise_type !='clean':
            # Load human noisy labels
            all_noisy_labels = self.load_label()
            self.all_noisy_labels = all_noisy_labels.tolist()
            print(f'noisy labels loaded from {self.noise_path}')

            if not is_human:
                T = np.zeros((self.nb_classes,self.nb_classes))
                for i in range(len(self.all_noisy_labels)):
                    T[self.all_labels[i]][self.all_noisy_labels[i]] += 1
                T = T/np.sum(T,axis=1)
                print(f'Noise transition matrix is \n{T}')
                all_noisy_labels = multiclass_noisify(y=np.array(self.all_labels), P=T,
                                    random_state=0) #np.random.randint(1,10086)
                self.all_noisy_labels = all_noisy_labels.tolist()
                T = np.zeros((self.nb_classes,self.nb_classes))
                for i in range(len(self.all_noisy_labels)):
                    T[self.all_labels[i]][self.all_noisy_labels[i]] += 1
                T = T/np.sum(T,axis=1)
                print(f'New synthetic noise transition matrix is \n{T}')

            self.all_labels = np.array(self.all_labels)
            self.all_noisy_labels = np.array(self.all_noisy_labels)
            self.select_dict = select_dict
            self.resplit()

            for i in range(len(self.all_noisy_labels)):
                idx_each_class_noisy[self.all_noisy_labels[i]].append(i)
            class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
            self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
            print(f'The noisy data ratio in each class is {self.noise_prior}')
            self.noise_or_not = np.transpose(self.all_noisy_labels)!=np.transpose(self.all_labels)
            self.actual_noise_rate = np.sum(self.noise_or_not)/self.noise_or_not.shape[0]
            print('over all noise rate is ', self.actual_noise_rate)

        
        

    def long_tail(self, data,tail_rate,dataset = 'cifar10'):
        long_tail_file = f'longtail_idx_{dataset}_tailrate_{tail_rate}.csv'
        longtial = np.genfromtxt(long_tail_file, delimiter=',',dtype=int)[1:,1]
        trimed_data = np.intersect1d(data,longtial)
        print("Resplit and long tail")
        return trimed_data

    def resplit(self):
        if self.select_dict =={}:
            # print("resplit!")
            
            self.train_select, self.test_select = self.resplit_idx(self.all_data, 0.8)
            self.select_dict['train_select'] = self.train_select
            self.select_dict['test_select'] = self.test_select

            # with open('train_resplit.npy', 'wb') as f:
            #     np.save(f, self.train_select)

        # from os.path import exists
        # if (exists("train_resplit.npy")):
        #     data = np.load('train_resplit.npy')
        #     assert torch.sum(torch.tensor(data) - torch.tensor(self.train_select)) == 0 

        # TODO: Make it in args
        long_tail_proc = False
        if self.train:
            # long tial process
            if long_tail_proc: self.select_dict['train_select'] = self.long_tail(self.select_dict['train_select'],'10.0')

            self.train_data = self.all_data = self.all_data[self.select_dict['train_select']]
            self.train_labels = self.all_labels = self.all_labels[self.select_dict['train_select']]
            self.train_noisy_labels = self.all_noisy_labels = self.all_noisy_labels[self.select_dict['train_select']]

            self.train_noise_label_dict = {}
            for key in self.noise_label_dict:
                self.train_noise_label_dict[key] = self.noise_label_dict[key][self.select_dict['train_select']]
        else:
            self.test_data = self.all_data = self.all_data[self.select_dict['test_select']]
            self.test_labels = self.all_labels = self.all_labels[self.select_dict['test_select']]
            self.test_noisy_labels = self.all_noisy_labels = self.all_noisy_labels[self.select_dict['test_select']]


    def resplit_idx(self, data, ratio):
        max_index  =data.shape[0]
        split_index = int(max_index*ratio)
        index = np.arange(start=0, stop=max_index, step=1)
        select = np.random.shuffle(index)
        select = index[:split_index]
        no_select = index[split_index:]
        return select,no_select


        return data[select] , data[no_select],select,no_select


    def load_label(self):
        #NOTE only load manual training label
        noise_label = torch.load(self.noise_path)
        self.noise_label_dict = noise_label
        if isinstance(noise_label, dict):
            if "clean_label" in noise_label.keys():
                clean_label = torch.tensor(noise_label['clean_label'])
                assert torch.sum(torch.tensor(self.all_labels) - clean_label) == 0  
                print(f'Loaded {self.noise_type} from {self.noise_path}.')
                print(f'The overall noise rate is {1-np.mean(clean_label.numpy() == noise_label[self.noise_type])}')

            if self.remix : 
                return self.mix_noise(noise_label).reshape(-1) 
            return noise_label[self.noise_type].reshape(-1)  
        else:
            raise Exception('Input Error')

    def mix_noise(self,noise_label):
        print("Re-Mixing noisey label: Rand1, Rand2, Rand3")
        rand1 = noise_label['random_label1']
        rand2 = noise_label['random_label2']
        rand3 = noise_label['random_label3']
        all_rand = np.stack([rand1,rand2,rand3])
        np.random.shuffle(all_rand)
    
        return all_rand[0,:]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target_clean, target_noisy = self.train_data[index], self.train_labels[index],self.train_noisy_labels[index]

            # noise_names = ['random_label1','random_label2','random_label3']
            # target_noisy = self.train_noise_label_dict[noise_names[random.randint(0,len(noise_names)-1)]][index]
            # if self.noise_type !='clean':
            #     img, target_clean, target_noisy = self.train_data[index], self.test_labels[index],self.train_noisy_labels[index]
            # else:
            #     img, target_clean, target_noisy = self.train_data[index], self.train_labels[index]
        else:
            # img, target = self.test_data[index], self.test_labels[index]
            img, target_clean, target_noisy = self.test_data[index], self.test_labels[index],self.test_noisy_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            # target = self.target_transform(target)
            target_clean, target_noisy = self.target_transform(target_clean), self.target_transform(target_noisy)

        if self.train:
            return img, target_noisy , index
        else:
            return img, target_clean , index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        self.train_list = [
            ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
            ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
            ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
            ['data_batch_4', '634d18415352ddfa80567beed471001a'],
            ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
        ]

        self.test_list = [
            ['test_batch', '40351d587109b95175f43aff81a1287e'],
        ]
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
