#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Module Docstring
"""
Boltzmann Machine Training
"""

#
# Imports
#

import ast
import cPickle                              as pkl
import cStringIO
import cv2
import getopt
import gzip
import h5py                                 as H
import inspect
import io
import math
import numpy                                as np
import os
import pdb
import sys
import tarfile
import theano                               as T
import theano.tensor                        as TT
import theano.tensor.nnet                   as TTN
import theano.tensor.nnet.conv              as TTNC
import theano.tensor.nnet.bn                as TTNB
import theano.tensor.signal.pool            as TTSP
from   theano import config                 as TC
import theano.printing                      as TP
import time
import traceback


###############################################################################
# Dummy object class
#

class Object(object): pass


###############################################################################
# Utilities
#


###############################################################################
# Implementation
#

class BM(Object):
	def __init__(self, config=[2, 1024]):
		"""Construct a Boltzmann Machine."""
		self.depth  = len(config)
		self.config = config
		self.theta  = BMTheta(self.config)
	
	def train(self, D, learnRate=0.001, numEpochs=20):
		"""Train the Boltzmann Machine on the given dataset. The dataset should be a
		numpy array whose every row has config[0] columns corresponding to attributes."""
		
		# Compute overall data variance
		dataVariance = self.computeDataVariance(D)
		
		# Iterate
		for e in xrange(numEpochs):
			for i in xrange(len(D)):
				# Temperature computation.
				temperature1Variance = self.getTemperature1Variance()
				Tmax                 = dataVariance / temperature1Variance
				K                    = int(np.ceil(np.log2(Tmax)))
				
				# Positive phase. Feedforward initialization of hiddens.
				minibatchOfVisibles  = D[i:i+1] # For now, just 1 example at a time.
				s                    = self.initStateFromVisibles(minibatchOfVisibles)
				
				# Negative phase.
				T                    = 1.0
				L                    = 0.0
				for _ in xrange(1,K):
					sNext  = self.sampleNextStateGivenCurrentState(s, T=T)
					L     += self.logProbability(sNext, s, T=T)
					self.updateNegativePhase(sNext, s, T=T)
					T     *= 2
					s      = sNext
				
				# Update convergence criterion
				L                   += self.logMarginalProbability(s)
		# Iterate until some convergence criterion on L is met.
	
	def computeDataVariance(self, D):
		"""Return the largest single variance amongst all the visibles."""
		var      = np.var(D, axis=0, keepdims=True)
		return np.max(var)
	
	def getTemperature1Variance(self):
		"""The largest variance due to the noise injected by the transition operator on the
		visibles at temperature 1 corresponds to the magnitude of the row in the weight matrix
		connecting the visibles to the first hidden layer with the highest magnitude."""
		
		W   = self.theta.W[0]
		mag = np.sqrt(np.sum(W**2, axis=1))
		return np.max(mag)
	
	def initStateFromVisibles(self, v):
		"""Initializes a minibatch of states from a minibatch of visibles, using purely feed-
		forward computation. The state is a list of 2D arrays; The first dimension of each of
		these arrays corresponds to the number of examples in the minibatch."""
		
		s = []
		s.append(v)
		
		h = v
		for i in xrange(1, self.depth):
			h = self.theta.B[i][np.newaxis,:] + np.dot(self.rho(h), self.theta.W[i-1])
			s.append(h)
		
		return s
	
	def sampleNextStateGivenCurrentState(self, s, T):
		sNext = s[:]
		
		# Resample odd-numbered layers
		i = 1
		while i<self.depth:
			# Check if this is the last layer (none above)
			lastLayer = i == self.depth-1
			
			# Resample
			Z = np.random.normal(size=sNext[i].shape)
			if lastLayer:
				W = self.theta.W[i]
				w = sNext[i-1]
				b = self.theta.B[i]
				
				sNext[i] = b + w.dot(W) + T*Z
			else:
				W = self.theta.W[i]
				w = sNext[i-1]
				V = self.theta.W[i+1]
				v = sNext[i+1]
				b = self.theta.B[i]
				
				sNext[i] = b + w.dot(W) + v.dot(V.T) + T*Z
			
			# Move to next odd layer
			i+=2
		
		# Resample even-numbered layers
		i = 0
		while i<self.depth:
			# Check if this is the first or last layer (none below/above)
			firstLayer = (i ==            0)
			lastLayer  = (i == self.depth-1)
			
			# Resample
			Z = np.random.normal(size=sNext[i].shape)
			if firstLayer:
				V = self.theta.W[i+1]
				v = sNext[i+1]
				b = self.theta.B[i]
				
				sNext[i] = b + v.dot(V.T) + T*Z
			elif lastLayer:
				W = self.theta.W[i]
				w = sNext[i-1]
				b = self.theta.B[i]
				
				sNext[i] = b + w.dot(W) + T*Z
			else:
				W = self.theta.W[i]
				w = sNext[i-1]
				V = self.theta.W[i+1]
				v = sNext[i+1]
				b = self.theta.B[i]
				
				sNext[i] = b + w.dot(W) + v.dot(V.T) + T*Z
			
			# Move to next even layer
			i+=2
	
	def logProbability(sNext, s, T):
		return 0.0
	
	def updateNegativePhase(self,s,sp,T):
		pass
	
	def logMarginalProbability(sNext, s):
		return 0.0
	
	def rho(self, x):
		"""Non-linearity. It was proposed to use rect(tanh(x))."""
		return np.maximum(0, np.tanh(x))


class BMTheta(Object):
	def __init__(self, config=[2,1024]):
		"""Initialize a BM theta object. A BM theta contains the parameters (biases and weights) of the
		Boltzmann machine."""
		self.depth  = len(config)
		self.config = config
		self.B      = []
		self.W      = []
		
		# Create self.depth biases.
		for i in xrange(self.depth):
			self.B.append(np.zeros((self.config[i],), dtype="float32"))
		
		# Create self.depth-1 weigth matrices.
		for i in xrange(self.depth-1):
			fin   = self.config[i]
			fout  = self.config[i+1]
			scale = np.sqrt(2.0/(fin+fout))
			self.W.append(np.random.normal(scale=scale, size=(fin, fout)).astype("float32"))
	def clone(self):
		"""Clone (deep copy) a BM theta object."""
		newT = BMTheta(self.config)
		
		for dst,src in zip(newT.B, self.B):
			np.copyto(dst, src)
		for dst,src in zip(newT.W, self.W):
			np.copyto(dst, src)
		
		return newS







###############################################################################
# Implementations of the script's "verbs".
#

def verb_help(argv=None):
	"""
Usage of BMTrain.

The BMTrain script is invoked using a verb that denotes the general action to be
taken, plus optional arguments. The following verbs are defined:

-   help:
        This help message.

-   train:
        Train the Boltzmann Machine.

"""[1:-1]  #This hack deletes the newlines before and after the triple quotes.
	print(verb_help.__doc__)
def verb_train(argv=None):
	"""Train BM."""
	
	# FIXME: Generate a proper manifold here.
	D  = np.random.normal(loc=0.0, scale=1.0, size=(10000,2))
	bm = BM()
	bm.train(D)
	
	pdb.set_trace()
def verb_screw(argv=None):
	"""Screw around."""
	
	pass


###############################################################################
# Main
#

if __name__ == "__main__":
    #
    # This script is invoked using a verb that denotes the general action to
    # take, plus optional arguments. If no verb is provided or the one
    # provided does not match anything, print a help message.
    #
    
    if((len(sys.argv) > 1)                      and # Have a verb?
       ("verb_"+sys.argv[1] in globals())       and # Have symbol w/ this name?
       (callable(eval("verb_"+sys.argv[1])))):      # And it is callable?
        eval("verb_"+sys.argv[1])(sys.argv)        # Then call it.
    else:
        verb_help(sys.argv)                        # Or offer help.
