import numpy as np
import torch
import augmentations

def aug(image, preprocess): 
   """Perform AugMix augmentations and compute mixture.   
   Args: 
    image: PIL.Image input image 
    preprocess: Preprocessing function which should return a torch tensor.   
   Returns: 
     mixed: Augmented and mixed image. 
   """ 
   mixture_width = 3
   mixture_depth = -1
   aug_severity = 1
   ws = np.float32(np.random.dirichlet([1] * mixture_width)) 
   m = np.float32(np.random.beta(1, 1)) 

   mix = torch.zeros_like(preprocess(image)) 
   for i in range(mixture_width): 
     image_aug = image.copy() 
     depth = mixture_depth if mixture_depth > 0 else np.random.randint(1, 4) 
     for _ in range(depth): 
       op = np.random.choice(augmentations.augmentations) 
       image_aug = op(image_aug, aug_severity) 
     # Preprocessing commutes since all coefficients are convex 
     #  k个增广加权融合
     mix += ws[i] * preprocess(image_aug) 
   #  与原图加权融合
   mixed = (1 - m) * preprocess(image) + m * mix 
   return mixed 
