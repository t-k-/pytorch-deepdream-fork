import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops

def load_image(path):
    image = Image.open(path)
    plt.imshow(image)
    plt.title("Image loaded successfully")
    plt.pause(2)
    return image

# normalise = transforms.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225]
#     )

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()#,
    #normalise
    ])

# def deprocess(image):
#     return image * torch.Tensor([0.229, 0.224, 0.225]).cuda()  + torch.Tensor([0.485, 0.456, 0.406]).cuda()

vgg = models.vgg16(pretrained=True)
vgg = vgg.cuda()
print(vgg)
modulelist = list(vgg.features.modules())

def dd_helper(image, layer, iterations, lr):
    input = preprocess(image).unsqueeze(0).cuda()

    input = Variable(input, requires_grad=True)
    vgg.zero_grad()
    for i in range(iterations):
        # forward
        out = input
        for j in range(layer):
            out = modulelist[j+1](out)
        loss = out.norm()
        # backward
        loss.backward()
        # simple gradient descent
        input.data = input.data + lr * input.grad.data

    input = input.data.squeeze()
    input.transpose_(0,1)
    input.transpose_(1,2)
    # input = deprocess(input).cpu()
    input = input.cpu()
    input = np.clip(input, 0, 1)
    im = Image.fromarray(np.uint8(input*255))
    return im

def deep_dream_vgg(image, layer, iterations, lr, num_octaves):

    if num_octaves>0:
        image1= image.filter(ImageFilter.GaussianBlur(2))
        image1= deep_dream_vgg(image1, layer, iterations, lr, num_octaves-1)

    img_result = dd_helper(image, layer, iterations, lr)
    print('iteration', num_octaves)
    plt.imshow(img_result)
    plt.pause(1)
    return img_result

sky = load_image('sky-dd.jpeg')
sky_26 = deep_dream_vgg(sky, 26, 5, 0.2, 10)
