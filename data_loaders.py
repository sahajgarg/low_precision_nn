import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from transformers.data.data_collator import DataCollatorWithPadding

from models.dlrm.dlrm_data_pytorch import CriteoDataset, collate_wrapper_criteo

import torch
from torchvision import datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def get_dlrm_dataloaders(data_path, train_batch_size, val_batch_size):
    raw_file = os.path.join(data_path, 'train.txt')
    processed_file = os.path.join(data_path, 'kaggleAdDisplayChallenge_processed.npz')

    train_data = CriteoDataset('kaggle', -1, 0.875, 'total', "train", raw_file, processed_file, True)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=train_batch_size,
        shuffle=False,
        collate_fn=collate_wrapper_criteo,
        pin_memory=False,
        drop_last=False,  # True
    )

    test_data = CriteoDataset('kaggle', -1, 0.875, 'total', "test", raw_file, processed_file, True)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_wrapper_criteo,
        pin_memory=False,
        drop_last=False,  # True
    )

    return train_loader, test_loader


def get_imagenet_dataloaders(data_path, train_batch_size, val_batch_size, train_data_path, val_data_path, args, train_subset_size=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if train_data_path is None:
        traindir = os.path.join(data_path, 'imagenet/train_subset')
    else:
        traindir = train_data_path
    
    print("Training on {}".format(traindir))
    crop_size = 299 if args.model == "inceptionv3" else 224
    dataset_train = ImageFolderWithPaths(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))


    num_epochs = -1
    if train_subset_size is not None and train_subset_size != len(dataset_train): 
        full_batches = np.ceil(len(dataset_train)/train_batch_size)
        full_batches = full_batches if args.train_batches == -1 else min(full_batches, args.train_batches)
        subset_batches = np.ceil(train_subset_size/train_batch_size)
        subset_batches = subset_batches if args.train_batches == -1 else min(subset_batches, args.train_batches)
        num_epochs = int(np.ceil(full_batches/subset_batches))

        np.random.seed(1234)
        train_subset_indices = np.random.choice(len(dataset_train), train_subset_size)
        dataset_train = torch.utils.data.Subset(dataset_train, train_subset_indices) 
    
    print("Training with {} samples".format(len(dataset_train)))
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=train_batch_size, num_workers=4,
        shuffle=True, pin_memory=True, drop_last=False)

    if val_data_path is None:
        valdir = os.path.join(data_path, 'full_imagenet/val_2012/val_formatted')
    else:
        valdir = val_data_path
    
    print("Evaluating on {}".format(valdir))
    resize_size = 299 if args.model == "inceptionv3" else 256
    dataset_val = ImageFolderWithPaths(
        valdir,
        transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]))

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=val_batch_size, num_workers=4,
        shuffle=False, pin_memory=True, drop_last=False)

    return data_loader_train, data_loader_val, num_epochs


def get_glue_dataloaders(tokenizer, train_batch_size, val_batch_size):
    datasets = load_dataset("glue", "mnli")
    sentence1_key, sentence2_key = ("premise", "hypothesis")
    padding = "max_length"
    max_length = 128

    label_to_id = None

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True)

    train_dataset = datasets["train"]
    val_matched_dataset = datasets["validation_matched"]
    val_mismatched_dataset = datasets["validation_mismatched"]
    train_dataset.set_format(type=train_dataset.format["type"], columns=['attention_mask', 'input_ids', 'label', 'token_type_ids'])
    val_matched_dataset.set_format(type=val_matched_dataset.format["type"], columns=['attention_mask', 'input_ids', 'label', 'token_type_ids'])
    val_mismatched_dataset.set_format(type=val_matched_dataset.format["type"], columns=['attention_mask', 'input_ids', 'label', 'token_type_ids'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        collate_fn = DataCollatorWithPadding(tokenizer),
        shuffle=True,
        pin_memory=False,
        drop_last=False,  
    )

    test_loader_1 = torch.utils.data.DataLoader(
        val_matched_dataset,
        batch_size=val_batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer),
        shuffle=False,
        pin_memory=False,
        drop_last=False, 
    )

    test_loader_2 = torch.utils.data.DataLoader(
        val_mismatched_dataset,
        batch_size=val_batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer),
        shuffle=False,
        pin_memory=False,
        drop_last=False,  # True
    )

    return train_loader, (test_loader_1, test_loader_2)


def prepare_data_loaders(dataset, data_path, train_batch_size, val_batch_size, 
        train_data_path, val_data_path, args, train_subset_size=None, tokenizer=None):
    if dataset == 'criteo':
        train_loader, test_loader = get_dlrm_dataloaders(data_path, train_batch_size, val_batch_size)
        num_epochs = args.epochs
    elif dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_dataloaders(data_path, train_batch_size, val_batch_size)
        num_epochs = args.epochs
    elif dataset == 'mnli':
        train_loader, test_loader = get_glue_dataloaders(tokenizer, train_batch_size, val_batch_size)
        num_epochs = args.epochs
    else:
        train_loader, test_loader, num_epochs = get_imagenet_dataloaders(data_path, train_batch_size, 
                val_batch_size, train_data_path, val_data_path, args, train_subset_size)

    return train_loader, test_loader, num_epochs


def get_cifar10_dataloaders(data_path, train_batch_size, val_batch_size):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    dataset_train = torchvision.datasets.CIFAR10(root="data", train=True, 
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                transforms.Normalize(mean, std)]), download=True)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=train_batch_size, num_workers=4,
        shuffle=True, pin_memory=True, drop_last=False)

    dataset_val = torchvision.datasets.CIFAR10(root="data", train=False,
            transform=([transforms.ToTensor(), 
                transforms.Normalize(mean, std)]), download=True)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=val_batch_size, num_workers=4,
        shuffle=False, pin_memory=True, drop_last=False)

    return data_loader_train, data_loader_val

