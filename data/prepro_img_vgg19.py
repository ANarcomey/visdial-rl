import argparse
import os
import glob
import json
import numpy as np
from PIL import Image
import h5py
import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms



class ResnetModule(nn.Module):
    """
    Resnet Model without the fc layer
    """
    def __init__(self, delete_starting="fc"):
        super(ResnetModule, self).__init__()
        # create resnet
        #resnet = models.resnet50(pretrained=True)
        resnet = models.resnet152(pretrained=True)
        if delete_starting == "fc":
            modules = list(resnet.children())[:-1] # delete the last fc layer.
        elif delete_starting == "res3":
            modules = list(resnet.children())[:-3]
        elif delete_starting == "none":
            modules = list(resnet.children())[:]
        else:
            raise ValueError("Invalid deletion")
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)


def get_features(image_root, image_names, resnet_im, split):

    feats_all_images = []
    for i, im_name in enumerate(image_names):
        if i % 1000 == 0:
            print("Extracting features for {} image # {} out of {}."
                .format(split, i, len(image_names)))
        image = Image.open(os.path.join(image_root,im_name)).convert("RGB")
        image = np.asarray(image)

        # transform image
        im = Image.fromarray(image)
        im = transform(im)

        im = Variable(im.unsqueeze(0))
        im = im.cuda()
        feats = resnet_im(im).squeeze(0).cpu().data.numpy()

        feats_all_images.append(feats.flatten())
        
    return feats_all_images

def get_features_batched(image_root, image_names, resnet_im, split, batch_size, embed_size):
    feats_all_images = np.zeros([len(image_names), embed_size])
    for i_start, im_name_batch in batch_iter(image_names, batch_size):
        if i_start % 1024 == 0:
            print("Extracting features for {} image # {} out of {}."
                .format(split, i_start, len(image_names)))
        image_batch = [Image.open(os.path.join(image_root,im_name)).convert("RGB") for im_name in im_name_batch]
        #image = [np.asarray(image) for image in image_batch]

        # transform image
        #im = Image.fromarray(image)
        im_batch = [transform(im) for im in image_batch]

        im_batch = Variable(torch.stack(im_batch))
        im_batch = im_batch.cuda()
        feats_batch = resnet_im(im_batch).view(len(im_batch), embed_size)
        feats_batch = feats_batch.cpu().data.numpy()

        feats_all_images[i_start:i_start + len(feats_batch)] = feats_batch
        
    return feats_all_images

def batch_iter(image_names, batch_size):

    for i in range(0, len(image_names), batch_size):  
        #yield list(range(i,len(image_names[i:i + batch_size]))), image_names[i:i + batch_size] 
        yield i, image_names[i:i + batch_size] 


#l = [1,2,3,4,5]
#for indices, vals in batch_iter(l,2):
#    print(indices, vals)

# Parse arguments
parser = argparse.ArgumentParser(description='Extract features and encode in HDF5')
parser.add_argument('-imageRoot', type=str, help='Path to COCO image root')
parser.add_argument('-imageInfo', type=str, help='Path to image data info json')
parser.add_argument('-outputH5', type=str, help='H5 filename to write data')
parser.add_argument('-batchSize', default=64, help='Batch size for extracting features')
parser.add_argument('-embedSize', default=16384, help='Embedding size for each image')

args = parser.parse_args()

# Create ResNet model
#import pdb;pdb.set_trace()
resnet_im = ResnetModule(delete_starting="res3")
resnet_im = resnet_im.cuda()
resnet_im.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

# Load image file names
image_info = json.load(open(args.imageInfo, 'r'))
train_im_names = image_info['unique_img_train']
val_im_names = image_info['unique_img_val']
test_im_names = image_info['unique_img_test']

# Extract image features
#import pdb;pdb.set_trace()

train_feats_all_images = get_features_batched(args.imageRoot, train_im_names, resnet_im, "train", args.batchSize, args.embedSize)
val_feats_all_images = get_features_batched(args.imageRoot, val_im_names, resnet_im, "val", args.batchSize, args.embedSize)
test_feats_all_images = get_features_batched(args.imageRoot, test_im_names, resnet_im, "test", args.batchSize, args.embedSize)


# Save into HDF5
h5 = h5py.File(args.outputH5,'w')
h5.create_dataset("images_train",data=np.array(train_feats_all_images))
h5.create_dataset("images_val",data=np.array(val_feats_all_images))
h5.create_dataset("images_test",data=np.array(test_feats_all_images))




