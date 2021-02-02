from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
from utils import *

# Data transformation with augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Dataset
class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, path


class Shot_Dataset(Dataset):
    def __init__(self, root, txt, transform=None, shot_phase='many_shot'):
        self.shot_list = get_shot_list()[shot_phase]
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.classid2label, self.label2classid = map_classid_and_label(self.shot_list)
        with open(txt) as f:
            for line in f:
                classid = int(line.split()[1])
                if classid in self.shot_list:
                    self.img_path.append(os.path.join(root, line.split()[0]))
                    self.labels.append(self.classid2label[classid])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, path


class Curric_Dataset(Dataset):

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.init_instance_weight()


    def init_instance_weight(self):
        self.instance_weights = np.zeros(len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        weight = self.instance_weights[index]
        return sample, label, path, weight



# Load datasets
def load_data(data_root, dataset, phase, batch_size, sampler_dic=None, num_workers=4, test_open=False, shuffle=True, shot_phase='many_shot', curric=False):
    
    txt = './data/%s/%s_%s.txt'%(dataset, dataset, (phase if phase != 'train_plain' else 'train'))

    print('Loading data from %s' % (txt))

    if phase not in ['train', 'val']:
        transform = data_transforms['test']
    else:
        transform = data_transforms[phase]

    print('Use data transformation:', transform)

    if shot_phase is not None:
        set_ = Shot_Dataset(data_root, txt, transform, shot_phase=shot_phase)
    elif curric is True:
        set_ = Curric_Dataset(data_root, txt, transform)
    else:
        set_ = LT_Dataset(data_root, txt, transform)

    if phase == 'test' and test_open:
        open_txt = './data/%s/%s_open.txt'%(dataset, dataset)
        print('Testing with opensets from %s'%(open_txt))
        open_set_ = LT_Dataset('./data/%s/%s_open'%(dataset, dataset), open_txt, transform)
        set_ = ConcatDataset([set_, open_set_])

    if sampler_dic and phase == 'train':
        print('Using sampler.')
        print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                           sampler=sampler_dic['sampler'](set_, sampler_dic['num_samples_cls']),
                           num_workers=num_workers)
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        return DataLoader(dataset=set_, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)
        
    
    
