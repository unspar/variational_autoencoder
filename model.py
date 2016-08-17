import tensorflow as tf
import numpy as np
import scipy.special as scisp 
import img_dataset as img
import os
import time
import random


import png


def var_init(fan_in, fan_out, constant=1):
  """ Xavier initialization of network weights"""
  # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
  low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
  high = constant*np.sqrt(6.0/(fan_in + fan_out))
  return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class Autoencoder():
  
  save_dir = "save"
 
  #TODO- make these dimensions tie into the dataset class 
  x_dim = 28
  y_dim = 28
  img_size = x_dim*y_dim

  num_epochs = 500
  epoch_size =100

  hidden_size = 500
  encoded_size = 20

  num_layers = 2
  
  batch_size = 100
  
  grad_clip = 5

  learning_rate = 0.001
  decay_rate = 0.97



  def __init__(s, sample = False):
    '''
    Builds the tensorflow autoencoder
    '''
    s.dataset = img.ImgDataset("train-images.idx3-ubyte")
    tf.set_random_seed(0)

    if sample == True:
      s.bs = 1
    else:
      s.bs = s.batch_size 
    #alais to save linespace
    f = tf.float32
    #batch size images, and the serialized input data
    s.inp = tf.placeholder(tf.float32, shape=(None, s.img_size))
    ac_fun = tf.nn.softplus
    
    
    #TODO-
    #Question:
    #
    #How do I format data before training them?
    #this work seems to indicate that pre-formatting the data is very helpful
    #(starting with this representation would have saved time)
    # 
    #Architecture:
    #a wrapper class? which is responsible for managing interactions with the model
    #a model class which establishes with tensorflow
    #used to collect input images
    s.test_inp = tf.identity(s.inp)
    #weights for encoder 
    s.enc= {
      "h1" : tf.Variable(var_init(s.img_size, s.hidden_size)),
      "h1b": tf.Variable(tf.zeros([s.hidden_size], dtype=f)),
      "h2" : tf.Variable(var_init(s.hidden_size, s.hidden_size)),
      "h2b": tf.Variable(tf.zeros([s.hidden_size], dtype=f)),

      "xmu" : tf.Variable(var_init(s.hidden_size, s.encoded_size)),
      "xmub": tf.Variable(tf.zeros([s.encoded_size],dtype=f)),

      "xsig": tf.Variable(var_init(s.hidden_size, s.encoded_size )),
      "xsigb": tf.Variable(tf.zeros([s.encoded_size],dtype=f))
    }

    s.eh1 = ac_fun( tf.add(tf.matmul(s.inp,s.enc["h1"]), s.enc["h1b"]))
    s.eh2 = ac_fun( tf.add(tf.matmul(s.eh1, s.enc["h2"]), s.enc["h2b"]))
    #From hidden layer generate mu and sigma using ac_fun (? why this? to prevent negative values?)
    
    #mu is mean of gaussian
    #sigma is log standard deviation of the gaussian (? why lg?)
    #ac_fun taken from https://jmetzen.github.io/2015-11-27/vae.html
    s.ln_sigma_sq = tf.add(tf.matmul(s.eh2, s.enc["xsig"]), s.enc["xsigb"])
    s.mu =tf.add(tf.matmul(s.eh2, s.enc["xmu"]), s.enc["xmub"])
    
    #this randomness helps to make the prob model work 
    #it reduces the information encoded in the model (reduce overfitting)
    s.eps = tf.random_normal((s.bs, s.encoded_size ), 0, 1, dtype=f)
    
    #Now we sample from our latent space (randomness comes from eps
    # z = mu + sigma*epsilon
    # (?why exp and then sqrt is that to turn it back into sigma?)
    #NOTE - tf.mul vs tf.matmul
    
    s.encoded = tf.add(s.mu, tf.mul(tf.sqrt(tf.exp(s.ln_sigma_sq)), s.eps))
    #s.encoded =  ac_fun(tf.add(tf.matmul(s.eh2, s.enc["xmu"]), s.enc["xmub"]) )



    #weights for the decoder
    s.dec= {
      "h1" : tf.Variable(var_init(s.encoded_size, s.hidden_size)),
      "h1b": tf.Variable(tf.zeros([s.hidden_size],dtype=f)),
      "h2" : tf.Variable(var_init(s.hidden_size, s.hidden_size)),
      "h2b": tf.Variable(tf.zeros([s.hidden_size],dtype=f)),
      "out": tf.Variable(var_init(s.hidden_size, s.img_size)),
      "outb": tf.Variable(tf.zeros([s.img_size],dtype=f))
    }
    s.dh1 = ac_fun(tf.add(tf.matmul(s.encoded, s.dec["h1"]), s.dec["h1b"]))
    s.dh2 = ac_fun(tf.add(tf.matmul(s.dh1, s.dec["h2"]), s.dec["h2b"]))
    
    #output image
    #sigmoid here?
    #adding a sigmoid here makes the input and output of identical forms.
    #is this valuable? 
    s.output =tf.nn.sigmoid((tf.add(tf.matmul(s.dh2, s.dec["out"]), s.dec["outb"])))
    
    #TODO- Undersand this
    # The loss is composed of two terms:
    # 1.) The reconstruction loss (the negative log probability
    #     of the input under the reconstructed Bernoulli distribution 
    #     induced by the decoder in the data space).
    #     This can be interpreted as the number of "nats" required
    #     for reconstructing the input when the activation in latent
    #     is given.
    # Adding 1e-10 to avoid evaluatio of log(0.0)
    #s.reconstr_loss = -tf.reduce_sum((s.inp * tf.log(1e-10 + s.output)) + ((1-s.inp) * tf.log(1e-10 + 1 - s.output)), 1)
    s.reconstr_loss = -tf.reduce_sum(s.inp * tf.log(1e-10 + s.output) + (1-s.inp) * tf.log(1e-10 + 1 - s.output),1)
    # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
   ##    between the distribution in latent space induced by the encoder on 
    #     the data and some prior. This acts as a kind of regularizer.
    #     This can be interpreted as the number of "nats" required
    #     for transmitting the the latent space distribution given
    #     the prior.

    #s.latent_loss = -0.5 * tf.reduce_sum(1 + s.log_sig - tf.square(s.mu) - tf.exp(s.log_sig), 1)
    #s.kl_divergence = -0.5 * tf.reduce_sum(1+ s.log_sig_sq - tf.square(s.mu) - tf.exp(s.log_sig_sq), 1) 
    s.latent_loss = -0.5 * tf.reduce_sum(1 + s.ln_sigma_sq - tf.square(s.mu)- tf.exp(s.ln_sigma_sq), 1)

    #s.cost = 0.5* tf.reduce_mean(tf.pow(s.inp - s.output, 2)) 
    #s.cost = tf.reduce_mean(s.kl_divergence + s.reconstr_loss)   # average over batch
    s.cost = tf.reduce_mean(s.reconstr_loss + s.latent_loss)   # average over batch   

 
    #learning rate
    #s.lr = tf.Variable(0.0, trainable=False)
    #tvars = tf.trainable_variables()
    ##grads, _ = tf.clip_by_global_norm(tf.gradients(s.cost, tvars), s.grad_clip)
    #note, optomizer.minimize also could work here
    s.optimizer = tf.train.AdamOptimizer(learning_rate=s.learning_rate).minimize(s.cost)
    #s.train_op = s.optimizer.apply_gradients(zip(grads, tvars)) 
    #s.optimizer = tf.train.AdamOptimizer(learning_rate=s.learning_rate).minimize(s.cost)
 
  def train(s):

    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      saver = tf.train.Saver(tf.all_variables())
      #TODO- work out hwo to restore a model from a save
      # restore model (if applicable)
      # saver.restore(sess, ckpt.model_checkpoint_path)
      for e in range(s.num_epochs):
        #sess.run(tf.assign(s.lr, s.learning_rate* s.decay_rate*e))
        s.dataset.reset()    
         
        save_dir = os.path.join(s.save_dir, 'model.ckpt')
        saver.save(sess,save_dir, global_step = e) 
        print("saving to " +s.save_dir)
        for n in range(s.epoch_size):
          start = time.time()  
          
          #this clip is important to normalize the data before processing
          feed = {s.inp: np.clip(s.dataset.readn(s.bs), 0, 1) }
          _, train_loss, hidden = sess.run([s.optimizer, s.cost, s.reconstr_loss], feed)
          end = time.time()
          print("batch: {}, epoch: {}, train_loss = {:.3f}, time/batch = {:.3f}" \
                   .format(e * s.epoch_size + n,
                         e, train_loss, end - start))


 
  def sample(s):
    '''samples from the latent space''' 
    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      saver = tf.train.Saver(tf.all_variables())
      
      ckpt = tf.train.get_checkpoint_state(s.save_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print("loaded model from : " + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path) 

      if s.bs != 1 : 
        raise ValueError("can only sample from models with a batch size and sequence length of one" )
      
      burn = s.dataset.readn(random.randint(1, 100) )
      img = np.clip(s.dataset.readn(1),0,1)
      feed = {s.inp:img}
      inp, output = sess.run([s.test_inp, s.output], feed)
      i = np.multiply(255, np.reshape(inp,(28,28))).astype('uint8')
      o = np.multiply(255, np.reshape(output,(28,28))).astype('uint8')
      
      png.from_array(i,'L').save('infile.png')
      png.from_array(o,'L').save('outfile.png')





 
