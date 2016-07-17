'''
convienience wrapper class for the minst data
'''

import idx2numpy as idx
import numpy as np

import scipy.misc

import tensorflow as tf


def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

class ImgDataset():
  '''
  returns the dataset object
  '''

  def __init__(s, filename):
    '''
    initializes the dataset
    '''  
    with tf.gfile.Open(filename, 'rb') as  bytestream:
      magic = _read32(bytestream)
      #TODO-work this out for an arbitrary dimensional file
      if magic != 2051:
        raise ValueError(
            'Invalid magic number %d in MNIST image file: %s' %
            (magic, filename))
      num_images = _read32(bytestream)
      rows = _read32(bytestream)
      cols = _read32(bytestream)
      buf = bytestream.read(rows * cols * num_images)
      data = np.frombuffer(buf, dtype=np.uint8)
      s.data = data.reshape(num_images, rows* cols).astype(float)
      s.pointer = 0 #index for reading from the dataset


  
   
  
  ''' 
  def extract_labels(filename, one_hot=False, num_classes=10):
    Extract the labels into a 1D uint8 numpy array [index]x
    print('Extracting', filename)
    with tf.gfile.Open(filename, 'rb') as  bytestream:
      magic = _read32(bytestream)
      if magic != 2049:
        raise ValueError(
            'Invalid magic number %d in MNIST label file: %s' %
            (magic, filename))
      num_items = _read32(bytestream)
      buf = bytestream.read(num_items)
      labels = np.frombuffer(buf, dtype=np.uint8)
      if one_hot:
        return dense_to_one_hot(labels, num_classes)
      return labels
  ''' 

  def readn(s,n):
    '''
    returns a numpy matrix of n X length dimensions 
    where length is the size of the image
    ''' 
    #TODO - handle overruns (if i call n > dataset length)
    ret = np.copy(s.data[s.pointer:s.pointer+n])
    s.pointer = s.pointer + n    
    return ret

  def reset(s):
    '''
    resets the pointer (intended for call on epoch completion)
    '''
    s.pointer = 0
    return

