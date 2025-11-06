# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2025/10/29 MBHImageMaskDatasetGenerator.py

import os

import shutil
import glob
import nibabel as nib
import numpy as np

import traceback
import cv2

class MBHImageMaskDatasetGenerator:

  def __init__(self, 
               images_dir  = "./", 
               masks_dir   = "./",
               output_dir = "./master", 
               resize     = 512):    
    self.images_dir = images_dir 
    self.masks_dir  = masks_dir

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    self.output_images_dir = os.path.join(output_dir, "images")
    self.output_masks_dir  = os.path.join(output_dir, "masks")
    os.makedirs(self.output_images_dir)
    os.makedirs(self.output_masks_dir)

    self.angle   = cv2.ROTATE_90_COUNTERCLOCKWISE
    self.RESIZE    = (resize, resize)
    self.file_format= ".png"

  def generate(self):
    index = 100000
    image_files = glob.glob(self.images_dir + "/*.nii.gz")
    image_files = sorted(image_files)

    mask_files = glob.glob(self.masks_dir   + "/*.nii.gz")
    mask_files = sorted(mask_files)   
    l1 = len(mask_files)
    l2 = len(image_files) 

    print("--- l1: {} l2: {}".format(l1, l2))
    #input("Hit any key.")
    if l1 != l2:
      raise Exception("Unmatched number of mask_files and image_files ")
    for i in range(l1):
      index += 1000
      self.generate_mask_files(mask_files[i],   index) 
      self.generate_image_files(image_files[i],index) 

  def normalize(self, image):
    min = np.min(image)/255.0
    max = np.max(image)/255.0
    scale = (max - min)
    if scale == 0:
      scale +=  1
    image = (image -min) / scale
    image = image.astype('uint8') 
    return image
  
  def generate_image_files(self, niigz_file, index):
    nii = nib.load(niigz_file)
    fdata  = nii.get_fdata()
    w, h, d = fdata.shape
    print("=== image shape {}".format(fdata.shape))
    for i in range(d):
      img = fdata[:,:, i]
      filename  = str(index + i) + self.file_format
      filepath  = os.path.join(self.output_images_dir, filename)
      corresponding_mask_file = os.path.join(self.output_masks_dir, filename)
      if os.path.exists(corresponding_mask_file):
        #img   = self.normalize(img)   
        img = cv2.resize(img, self.RESIZE)
        img = cv2.rotate(img, self.angle)
        cv2.imwrite(filepath, img)
        print("=== Saved {}".format(filepath))

      else:
        print("=== Skipped {}".format(filepath))

  def generate_mask_files(self, niigz_file, index ):
    nii = nib.load(niigz_file)
    fdata  = nii.get_fdata()
    w, h, d = fdata.shape
    print("=== mask shape {}".format(fdata.shape))
    for i in range(d):
      img = fdata[:,:, i]
      filename  = str(index + i) + self.file_format
      filepath  = os.path.join(self.output_masks_dir, filename)
      #Exclude all black empty mask
      if img.any() >0:
        img = cv2.resize(img, self.RESIZE)
        img = cv2.rotate(img, self.angle)
        img = self.colorize_mask(img)     
        cv2.imwrite(filepath, img)

      else:
        print("=== Skipped {}".format(filepath))

  def colorize_mask(self, mask):
    shp = mask.shape
    colorized = np.zeros((shp[0], shp[1], 3), dtype=np.float32)
    #            BGR triplet
    EDH = (  0,   0,  255)  #red
    IPH = (255,   0,   0)   #blue
    IVH = (  0, 255,  255)  #yellow
    SAH = (255, 255,   0)   #cyan
    SDH = (  0, 255,   0)   #green
    
    colorized[np.equal(mask, 1)] = EDH
    colorized[np.equal(mask, 2)] = IPH
    colorized[np.equal(mask, 3)] = IVH
    colorized[np.equal(mask, 4)] = SAH
    colorized[np.equal(mask, 5)] = SDH
    
    return colorized


if __name__ == "__main__":
  try:
    images_dir  = "./label_192/images/"
    masks_dir   = "./label_192/ground truths/"
    output_dir  = "./MBH-master/"
    generator = MBHImageMaskDatasetGenerator(images_dir = images_dir, 
                                            masks_dir   = masks_dir,
                                            output_dir  = output_dir)
    generator.generate()
  except:
    traceback.print_exc()