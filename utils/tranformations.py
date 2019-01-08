from imgaug import augmenters as iaa
import numpy as np



# class ImgAugTransform:
#   def __init__(self, ):
#     self.aug = iaa.Sequential([
#         iaa.Scale((224, 224)),
#         iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
#         iaa.Fliplr(0.5),
#         iaa.Affine(rotate=(-20, 20), mode='symmetric'),
#         iaa.Sometimes(0.25,
#                       iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
#                                  iaa.CoarseDropout(0.1, size_percent=0.5)])),
#         iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
#     ])
      
#   def __call__(self, img):
#     img = np.array(img)
#     return self.aug.augment_image(img)




class ImgAugTransform(object):
    def __init__(self, image_transform_list):
        self.image_transform = image_transform_list
        self.count = 0

    def __call__(self, img):
        img = np.array(img)
        return self.image_transform.augment_image(img)
