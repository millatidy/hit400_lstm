import sys
import os
import time
import numpy as np
from util import *
from datetime import datetime
from models.TheanoGRU import TheanoGRU

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "2000"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "48"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
NEPOCH = int(os.environ.get("NEPOCH", "20"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./dataset/test.txt")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "25000"))

if not MODEL_OUTPUT_FILE:
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
  MODEL_OUTPUT_FILE = "GRU-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

# Load data
x_train, y_train = load_data(INPUT_DATA_FILE)

# Build model
model = TheanoGRU(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM)

# Print SGD step time
t1 = time.time()
model.sgd_step(x_train[0], y_train[0], LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
sys.stdout.flush()

# We do this every few examples to understand what's going on
def sgd_callback(model, num_examples_seen):
  dt = datetime.now().isoformat()
  loss = model.calculate_loss(x_train[:10000], y_train[:10000])
  print("\n%s (%d)" % (dt, num_examples_seen))
  print("--------------------------------------------------")
  print("Loss: %f" % loss)
  generate_sentences(model, 10, index_to_word, word_to_index)
  save_model_parameters_theano(model, MODEL_OUTPUT_FILE)
  print("\n")
  sys.stdout.flush()

for epoch in range(NEPOCH):
  train_model(model, x_train, y_train, learning_rate=LEARNING_RATE, nepoch=1, decay=0.9,
    callback_every=PRINT_EVERY, callback=sgd_callback)
