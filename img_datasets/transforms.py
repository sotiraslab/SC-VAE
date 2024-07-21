import torchvision.transforms as transforms

def create_transforms(config, split='train', is_eval=False):
    if config.transforms.type == 'imagenet256x256':
        if split == 'train' and not is_eval:
            transforms_ = [
                transforms.Resize(256),
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        else:
            transforms_ = [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
    elif 'ffhq' in config.transforms.type:
        resolution = int(config.transforms.type.split('_')[0].split('x')[-1])
        if split == 'train' and not is_eval:
            transforms_ = [
                transforms.RandomResizedCrop(resolution, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        else:
            transforms_ = [
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
    elif config.transforms.type in ['LSUN', 'LSUN-cat', 'LSUN-church', 'LSUN-bedroom', 'LSUN-classroom']:
        resolution = 256 # only 256 resolution is supoorted for LSUN
        transforms_ = [
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    elif config.transforms.type in ['caltech256', 'caltech101']:
        resolution = 256 # only 256 resolution is supoorted for LSUN
        transforms_ = [
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    elif config.transforms.type == 'none':
        transforms_ = []
    else:
        raise NotImplementedError('%s not implemented..' % config.transforms.type)

    transforms_ = transforms.Compose(transforms_)

    return transforms_
