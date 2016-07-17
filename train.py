'''
Import
'''


import idx2numpy as idx
import numpy as np

import scipy.misc
import img_dataset as ig
import tensorflow as tf
import model as md

print('beginning training')
minst = ig.ImgDataset("train-images.idx3-ubyte")

ds = md.Autoencoder()

ds.train()


print('training complete')


