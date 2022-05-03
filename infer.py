import os
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from gan_module import Generator


parser = ArgumentParser()
parser.add_argument('--image_dir', default='/Downloads/CACD_VS/', help='The image directory')


@torch.no_grad()
def main():
    args = parser.parse_args()
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) if
                   x.endswith('.png') or x.endswith('.jpg')]
    model = Generator(ngf=32, n_residual_blocks=9)
    ckpt = torch.load('pretrained_model/state_dict.pth', map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    fig, ax = plt.subplots(2, 3, figsize=(10, 5))
    random.shuffle(image_paths)
    for i in range(3):
        img = Image.open(image_paths[i])
        img = trans(img).unsqueeze(0)
        aged_face = model(img)
        aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
        ax[0, i].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
        ax[1, i].imshow(aged_face)
    plt.show()

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
test_image = "/Users/paulfentress/Desktop/Mentia_Gans_Data.py/267.jpg"


test_image = Image.open(test_image)
beibs = np.array(Image.open("/Users/paulfentress/Desktop/Mentia_Gans_Data.py/beibs.png"))
print(type(test_image), np.array(test_image).shape)
test_image = test_image.resize((512,512))
test_image = np.array(test_image).reshape((1,512,512))
test_image = np.stack((test_image,test_image,test_image)).reshape((3,512,512))
print(type(test_image), test_image.shape)
print(type(beibs), beibs.shape)
print()


def reshape_image(image_path:str):
    """
    Arguments:
        image_path (string)
    returns:
        reshaped image of shape (3,512,512) (array)
    """

    # Here I am reading in the image, resizing the image using PIL because
    # the images are 48x48 squares, which are not multiple of 512x512.
    # Then once they are 512x512 I use numpy reshape to make them 1x512x512
    img = np.array((Image.open(image_path)).resize((512,512))).reshape((1,512,512))

    # Then once the img is shape 1x512x512 I stack the images ontop of eachother
    # using a numpy stack which results in a (3,1,512,512) shape, and then reshape
    # it again using numpy.
    img = np.stack((img,img,img)).reshape((512,512,3))
    return img

x = reshape_image("/Users/paulfentress/Desktop/Mentia_Gans_Data.py/267.jpg")






show = plt.imshow(x)
plt.show()

if __name__ == '__main__':
    main()
