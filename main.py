import gpt_2_simple as gpt2
from perplexity import get_perplexity
import os
import requests
import re

#------------------------
# Download model weights
#------------------------
def get_init():
    '''
    model_name (string): either "117M" or "124M". 124M indicates GPT2-small with 124 Mil parameters.
    purpose: downloads the trained model weights, hyperparameters and vocabulary (as Byte Pair Encodings)
    target: ./models in the current directory
    '''
    model_name = "124M"
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)

#------------------------
# Load the dataset as txt
#------------------------
def load_data(dataset, part='A'):
    '''
    inputs: datasets (string) - corresponds to name of the subfolder in the data directory, e.g. 'econ' or 'imdb'
            part (string) - identifies what part of the whole dataset is needed, if dataset is split (e.g. news)
    returns: file path to the dataset (or part of it)
    '''
    if dataset == 'news':
        basic = './data/' + dataset
        parts = {'A': '/data.txt', 'B': '/dataB.txt', 'C': '/dataC.txt'}
        directory = basic + parts[part]
    else:
        directory = './data/' + dataset + '/data.txt'
    return directory

#------------------------
# Get list of datasets
#------------------------
def list_data():
    '''
    purpose: get names of all datasets in the data subdirectory
    returns: list of folders in ./data
    '''
    directory = './data/'
    files = os.listdir(directory)
    return files


def clean(text):
    '''
    inputs: text (string) - GPT2 outputs that contain leftover artifacts from the training data, like '\n\n'
    purpose: remove all such artefacts
    returns: cleaned text file (string)
    '''
    out = text[0]
    out = re.sub(r'(\xa0a(7)?|\n)', '', out)
    out = re.sub(r'  ', ' ', out)
    return out

#------------------------
# Finetuning the model
#------------------------
sess = gpt2.start_tf_sess()     # Initialises a tensorflow session to track information
                                # do not turn this off, ever.
                                
gpt2.finetune(sess, dataset = load_data('news', part='C'),   # choose any .txt file from the ./data directory

              model_name='124M',  # searches for model defined in ./models/model_name
              
              steps=150,        # number of epochs to train for
              
              combine=2000,     # combine training examples into mini-groups of this size
              
              batch_size=1,     # number of batches fed to GPU per training cycle
              
              learning_rate=0.0002,  # learning rate for ADAM optimizer in Keras

              run_name='run4',  # run denotes model variant determined by training data. Eg. 'run1' trained on 'econ' data

              restore_from='latest',    # latest - if training across multiple .txt files | fresh - if training a new model
              
              use_memory_saving_gradients=True,  # Required for the model to train on my GPU

              accumulate_gradients=1,

              sample_every=50,  # generate a text sample at interval of these many epochs

              sample_length=100,    # number of words in generated sample
                            
              save_every=20,    # Save to disk after these many epochs
            )


# ------------------------
# Generating new text
# ------------------------
def generate_text(sess, dataset, custom_prompt=None, l=100, s=12345):
    # TODO - ADD NEW PROMPTS TO TEST ALL FOUR MODELS PLEASE

    prompt_list = {'econ': ['We present a novel way to ',
                        'Recent developments in the economic literature suggest that '],
               'imdb': ['The acting in this movie was terrible. I ', 'The acting in this movie was not great. I '],
               'news': [],
               'jokes': []
               }

    runs = {'econ': 'run1', 'imdb': 'run2', 'jokes': 'run3', 'news': 'run4'}    # mapping of datasets to models
    run = runs[dataset]

    gpt2.load_gpt2(sess, run_name=run)    # get the model trained on the input dataset

    if custom_prompt is None:
        prompts = prompt_list[dataset]    # use my default list of prompts to generate text
    else:
        prompts = [custom_prompt]         # if custom prompt is given then generate text using it

    for prompt in prompts:
        single_text = gpt2.generate(sess, run_name=run, model_name='124M', length=l, prefix=prompt, seed=s, return_as_list=True)
        output = clean(single_text)
        print('\n', f'Prompt: {prompt}', '\n', f'Output: {output}', '\n')


# EXAMPLE OF TEXT GENERATION
generate_text(sess, 'imdb')
