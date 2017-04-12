import re
import sys
# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Read file and save to list
word_training_file = open('training_lines.txt', 'r')
sentences = []
for line in word_training_file:
    if line.strip():
        word_list = re.sub("[^\w]", " ", line).split()
        sentences.append(word_list)
# for i in range(len(sentences)):
#     print sentences[i]
# sys.exit()

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-2   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context_movie_lines"
model.save(model_name)

# Test model
# model.doesnt_match("man woman child kitchen".split())
# model.doesnt_match("france england germany berlin".split())
# model.doesnt_match("paris berlin london austria".split())
model.most_similar("man")
model.most_similar("queen")
model.most_similar("awful")
