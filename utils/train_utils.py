import torch_optimizer
from easydict import EasyDict as edict
from torch import optim
import torch
import pandas as pd
from models import mnist, cifar, imagenet
from torch.utils.data import DataLoader
from onedrivedownloader import download as dn
from torch.optim import SGD

from utils.data_loader import get_train_datalist, ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics, get_test_datalist
import torch.nn.functional as F
import kornia.augmentation as K
import torch.nn as nn
from torch import Tensor
from utils.my_augment import Kornia_Randaugment
from torchvision import transforms
from utils.data_loader import Preprocess 
from tqdm import tqdm

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

class DataAugmentation(nn.Module):

    def __init__(self, inp_size, mean, std) -> None:
        super().__init__()
        self.randaugmentation = Kornia_Randaugment()
        self.inp_size = inp_size
        self.mean = mean
        self.std = std

        additional_aug = self.randaugmentation.form_transforms()
        self.transforms = nn.Sequential(
            K.Resize(size = (inp_size,inp_size)),
            K.RandomCrop(size = (inp_size,inp_size)),
            K.RandomHorizontalFlip(p=1.0),
            *additional_aug,
            K.Normalize(mean, std)
            )
        #self.cutmix = K.RandomCutMix(p=0.5)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:

        additional_aug = self.randaugmentation.form_transforms()
        self.transforms = nn.Sequential(
            K.Resize(size = (self.inp_size, self.inp_size)),
            K.RandomCrop(size = (self.inp_size, self.inp_size)),
            K.RandomHorizontalFlip(p=1.0),
            *additional_aug,
            K.Normalize(self.mean, self.std)
            )

        ##### check transforms
        # print("self.transform")
        # print(self.transforms)

        x_out = self.transforms(x)  # BxCxHxW
        #x_out, _ = self.cutmix(x_out)
        return x_out



def get_transform(dataset, transform_list, gpu_transform, use_kornia=True):
    mean, std, n_classes, inp_size, _ = get_statistics(dataset=dataset)
    if use_kornia:
        train_transform = DataAugmentation(inp_size, mean, std)
    else:
        train_transform = []
        if "cutout" in transform_list:
            train_transform.append(Cutout(size=16))
            if gpu_transform:
                gpu_transform = False
                print("cutout not supported on GPU!")
        if "randaug" in transform_list:
            train_transform.append(transforms.RandAugment())
        if "autoaug" in transform_list:
            if hasattr(transform_list, 'AutoAugment'):
                if 'cifar' in dataset:
                    train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('cifar10')))
                elif 'imagenet' in dataset:
                    train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')))
            else:
                train_transform.append(select_autoaugment(dataset))
                gpu_transform = False
        if "trivaug" in transform_list:
            train_transform.append(transforms.TrivialAugmentWide())
        if gpu_transform:
            train_transform = transforms.Compose([
                transforms.RandomCrop(inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                *train_transform,
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((inp_size, inp_size)),
                    transforms.RandomCrop(inp_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    *train_transform,
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    print(f"Using train-transforms {train_transform}")

    test_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, test_transform

def select_optimizer(opt_name, lr, model):
    if "adam" in opt_name:
        params = [param for name, param in model.named_parameters() if 'fc' not in name]
        opt = optim.Adam(params, lr=lr, weight_decay=0)
    elif "sgd" in opt_name:
        params = [param for name, param in model.named_parameters() if 'fc' not in name]
        opt = optim.SGD(
            params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")
    if 'freeze_fc' not in opt_name:
        opt.add_param_group({'params': model.fc.parameters()})
    return opt

def select_scheduler(sched_name, opt, hparam=None):
    if "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2
        )
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            opt, milestones=[30, 60, 80, 90], gamma=0.1
        )
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    return scheduler



def get_ckpt_remote_url(pre_dataset):
    if pre_dataset == "cifar100":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21108&authkey=AFsCv4BR-bmTUII" width="98" height="120" frameborder="0" scrolling="no"></iframe>',"rs18_cifar100.pth"

    elif pre_dataset == "tinyimgR":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21106&authkey=AKTxp5LFQJ9z9Ok" width="98" height="120" frameborder="0" scrolling="no"></iframe>', "erace_pret_on_tinyr.pth"

    elif pre_dataset == "imagenet":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21107&authkey=ADHhbeg9cUoqJ0M" width="98" height="120" frameborder="0" scrolling="no"></iframe>',"rs50_imagenet_full.pth"

    else:
        raise ValueError("Unknown auxiliary dataset")


def load_initial_checkpoint(pre_dataset, model, device, load_cp_path = None):
    url, ckpt_name = get_ckpt_remote_url(pre_dataset)
    load_cp_path = load_cp_path if load_cp_path is not None else './checkpoints/'
    print("Downloading checkpoint file...")
    dn(url, load_cp_path)
    print(f"Downloaded in: {load_cp}")
    net = load_cp(load_cp_path, model, device, moco=True)
    print("Loaded!")
    return net

def generate_initial_checkpoint(net, pre_dataset, pre_epochs, num_aux_classes, device, opt_args):
    aux_dset, aux_test_dset = get_aux_dataset()
    net.fc = torch.nn.Linear(net.fc.in_features, num_aux_classes).to(device)
    net.train()
    opt = SGD(net.parameters(), lr=opt_args["lr"], weight_decay=opt_args["optim_wd"], momentum=opt_args["optim_mom"])
    sched = None
    if self.args.pre_dataset.startswith('cub'):
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[80, 150, 250], gamma=0.5)
    elif 'tinyimg' in self.args.pre_dataset.lower():
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[20, 30, 40, 45], gamma=0.5)

    for e in range(pre_epochs):
        for i, (x, y, _) in tqdm(enumerate(aux_dl), desc='Pre-training epoch {}'.format(e), leave=False, total=len(aux_dl)):
            y = y.long()
            opt.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            aux_out = net(x)
            aux_loss = loss(aux_out, y)
            aux_loss.backward()
            opt.step()

        if sched is not None:
            sched.step()
        if e % 5 == 4:
            print(e, f"{self.mini_eval()*100:.2f}%")
    from datetime import datetime
    # savwe the model
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    modelpath = "my_checkpoint" + '_' + now + '.pth'
    torch.save(net.state_dict(), modelpath)
    print(modelpath)

def load_cp(cp_path, net, device, moco=False) -> None:
    """
    Load pretrain checkpoint, optionally ignores and rebuilds final classifier.

    :param cp_path: path to checkpoint
    :param new_classes: ignore and rebuild classifier with size `new_classes`
    :param moco: if True, allow load checkpoint for Moco pretraining
    """
    print("net")
    print([name for name, _ in net.named_parameters()])
    s = torch.load(cp_path, map_location=device)
    print("s keys", s.keys())
    '''
    if 'state_dict' in s:  # loading moco checkpoint
        if not moco:
            raise Exception(
                'ERROR: Trying to load a Moco checkpoint without setting moco=True')
        s = {k.replace('encoder_q.', ''): i for k,
             i in s['state_dict'].items() if 'encoder_q' in k}
    '''

    #if not ignore_classifier: # online CL이므로 fc out-dim을 1부터 시작
    net.fc = torch.nn.Linear(
        net.fc.in_features, 1).to(device) # online이므로 num_aux_classes => 1

    for k in list(s):
        if 'fc' in k:
            s.pop(k)
    for k in list(s):
        if 'net' in k:
            s[k[4:]] = s.pop(k)
    for k in list(s):
        if 'wrappee.' in k:
            s[k.replace('wrappee.', '')] = s.pop(k)
    for k in list(s):
        if '_features' in k:
            s.pop(k)

    try:
        net.load_state_dict(s)
    except:
        _, unm = net.load_state_dict(s, strict=False)
        print("unm")
        print(unm)
        '''
        if new_classes is not None or ignore_classifier:
            assert all(['classifier' in k for k in unm]
                       ), f"Some of the keys not loaded where not classifier keys: {unm}"
        else:
            assert unm is None, f"Missing keys: {unm}"
        '''

    return net
'''
def partial_distill_loss(model, net_partial_features: list, pret_partial_features: list,
                         targets, teacher_forcing: list = None, extern_attention_maps: list = None):

    assert len(net_partial_features) == len(
        pret_partial_features), f"{len(net_partial_features)} - {len(pret_partial_features)}"

    if teacher_forcing is None or extern_attention_maps is None:
        assert teacher_forcing is None
        assert extern_attention_maps is None

    loss = 0
    attention_maps = []

    for i, (net_feat, pret_feat) in enumerate(zip(net_partial_features, pret_partial_features)):
        assert net_feat.shape == pret_feat.shape, f"{net_feat.shape} - {pret_feat.shape}"

        adapter = getattr(
            model, f"adapter_{i+1}")

        pret_feat = pret_feat.detach()

        if teacher_forcing is None:
            curr_teacher_forcing = torch.zeros(
                len(net_feat,)).bool().to(self.device)
            curr_ext_attention_map = torch.ones(
                (len(net_feat), adapter.c)).to(self.device)
        else:
            curr_teacher_forcing = teacher_forcing
            curr_ext_attention_map = torch.stack(
                [b[i] for b in extern_attention_maps], dim=0).float()

        adapt_loss, adapt_attention = adapter(net_feat, pret_feat, targets,
                                              teacher_forcing=curr_teacher_forcing, attention_map=curr_ext_attention_map)

        loss += adapt_loss
        attention_maps.append(adapt_attention.detach().cpu().clone().data)

    return loss / (i + 1), attention_maps
'''
def get_data_loader(opt_dict, dataset, pre_train=False):
    if pre_train:
        batch_size = 128
    else:
        batch_size = opt_dict['batchsize']

    # pre_dataset을 위한 dataset 불러오고 dataloader 생성
    train_transform, test_transform = get_transform(dataset, opt_dict['transforms'], opt_dict['gpu_transform'])

    test_datalist = get_test_datalist(dataset)
    train_datalist, cls_dict, cls_addition = get_train_datalist(dataset, opt_dict["sigma"], opt_dict["repeat"], opt_dict["init_cls"], opt_dict["rnd_seed"])

    # for debugging!
    # train_datalist = train_datalist[:2000]

    exp_train_df = pd.DataFrame(train_datalist)
    exp_test_df = pd.DataFrame(test_datalist)

    train_dataset = ImageDataset(
        exp_train_df,
        dataset=dataset,
        transform=train_transform,
        preload = True,
        use_kornia=True,
        #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
        data_dir=opt_dict["data_dir"]
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=opt_dict["n_worker"],
    )

    test_dataset = ImageDataset(
        exp_test_df,
        dataset=dataset,
        transform=test_transform,
        #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
        data_dir=opt_dict["data_dir"]
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,#opt_dict["batchsize"],
        num_workers=opt_dict["n_worker"],
    )

    return train_loader, test_loader

def select_model(model_name, dataset, num_classes=None, opt_dict=None):
    opt = edict(
        {
            "depth": 18,
            "num_classes": num_classes,
            "in_channels": 3,
            "bn": True,
            "normtype": "BatchNorm",
            "activetype": "ReLU",
            "pooltype": "MaxPool2d",
            "preact": False,
            "affine_bn": True,
            "bn_eps": 1e-6,
            "compression": 0.5,
        }
    )

    if "mnist" in dataset:
        model_class = getattr(mnist, "MLP")
    elif "cifar" in dataset:
        model_class = getattr(cifar, "ResNet")
    elif "imagenet" in dataset:
        model_class = getattr(imagenet, "ResNet")
    else:
        raise NotImplementedError(
            "Please select the appropriate datasets (mnist, cifar10, cifar100, imagenet)"
        )
    if model_name == "resnet18":
        opt["depth"] = 18
    elif model_name == "resnet32":
        opt["depth"] = 32
    elif model_name == "resnet34":
        opt["depth"] = 34
    elif model_name == "mlp400":
        opt["width"] = 400
    else:
        raise NotImplementedError(
            "Please choose the model name in [resnet18, resnet32, resnet34]"
        )

    model = model_class(opt)

    # TODO initial check
    initial = False

    # pre_dataset 설정
    pre_dataset = None
    pre_dataset_num_class = 0
    path_load_cp = None
    if dataset == "cifar10":
        pre_dataset = "cifar100"
        path_load_cp = "res18_cifar100_pretrained_model.pth" #"checkpoint/rs18_cifar100_new.pth"
        pre_dataset_num_class = 100
    elif dataset == "cifar100":
        pre_dataset = "tiny_imagenet"
        pre_dataset_num_class = 1000

    assert pre_dataset is not None # none이면 설정이 덜 된것
    if opt_dict is not None:

        if initial:
            device = opt_dict['device']

            '''
            # pre_dataset을 위한 dataset 불러오고 dataloader 생성
            train_transform, test_transform = get_transform(pre_dataset, opt_dict['transforms'], opt_dict['gpu_transform'])

            test_datalist = get_test_datalist(pre_dataset)
            train_datalist, cls_dict, cls_addition = get_train_datalist(pre_dataset, opt_dict["sigma"], opt_dict["repeat"], opt_dict["init_cls"], opt_dict["rnd_seed"])

            # for debugging!
            # train_datalist = train_datalist[:2000]

            exp_train_df = pd.DataFrame(train_datalist)
            exp_test_df = pd.DataFrame(test_datalist)

            train_dataset = ImageDataset(
                exp_train_df,
                dataset=pre_dataset,
                transform=train_transform,
                preload = True,
                use_kornia=True,
                #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
                data_dir=opt_dict["data_dir"]
            )
            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=128,
                num_workers=opt_dict["n_worker"],
            )

            test_dataset = ImageDataset(
                exp_test_df,
                dataset=pre_dataset,
                transform=test_transform,
                #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
                data_dir=opt_dict["data_dir"]
            )
            test_loader = DataLoader(
                test_dataset,
                shuffle=False,
                batch_size=128,#opt_dict["batchsize"],
                num_workers=opt_dict["n_worker"],
            )
            '''
            train_loader, test_loader = get_data_loader(opt_dict, pre_dataset, pre_train=True)

            model.fc = torch.nn.Linear(model.fc.in_features, pre_dataset_num_class).to(device)
            model.to(device)
            model.train()
            #opt = SGD(model.parameters(), lr=opt_dict["lr"], weight_decay=opt_dict["optim_wd"], momentum=opt_dict["optim_mom"])
            opt = optim.Adam(model.parameters(), lr=opt_dict["lr"])
            criterion = F.cross_entropy

            for epoch in range(opt_dict["pre_epoch"]):
            #for epoch in range(10):
                correct = 0
                num_data = 0
                total_loss = 0.0
                iteration = 0

                for i, data in tqdm(enumerate(train_loader), desc='Pre-training epoch {}'.format(epoch), leave=False, total=len(train_loader)):
                    model.train()
                    x = data["image"]
                    y = data["label"]
                    x = x.to(device)
                    y = y.to(device)
                    logit = model(x)
                    loss = criterion(logit, y)
                    loss.backward()
                    opt.step()

                    _, preds = logit.topk(1, 1, True, True)
                    correct += torch.sum(preds == y.unsqueeze(1)).item()
                    num_data += y.size(0)
                    total_loss += loss.item()
                    iteration+=1

                print(f"[TRAIN] epoch{epoch} loss",  total_loss / iteration, "accuracy", correct / num_data)

                if epoch % 10 == 0:
                    model.eval()
                    for i, data in enumerate(test_loader):
                        x = data["image"]
                        y = data["label"]
                        x = x.to(device)
                        y = y.to(device)
                        logit = model(x)
                        loss = criterion(logit, y)

                        _, preds = logit.topk(1, 1, True, True)
                        correct += torch.sum(preds == y.unsqueeze(1)).item()
                        num_data += y.size(0)
                        total_loss += loss.item()
                        iteration+=1

                    print("[TEST] epoch{epoch} loss",  total_loss / iteration, "accuracy", correct / num_data)

            torch.save(model.state_dict(), "res18_cifar100_pretrained_model.pth")        
        else:
            model = load_initial_checkpoint(pre_dataset, model, opt_dict["device"], load_cp_path = path_load_cp)
            
    return model
