import torch
from wavenet_model import *
from audio_data import WavenetDataset
from wavenet_training import *
from model_logging import *


# initialize cuda option
dtype = torch.FloatTensor # data type
ltype = torch.LongTensor # label type

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor

model = load_latest_model_from('snapshots', use_cuda=use_cuda)

print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())


data = WavenetDataset(dataset_file='train_samples/bach_chaconne/dataset.npz',
                      item_length=model.receptive_field + model.output_length - 1,
                      target_length=model.output_length,
                      file_location='train_samples/bach_chaconne',
                      test_stride=500)
print('the dataset has ' + str(len(data)) + ' items')
cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")
start_data = data[250000][0] # use start data from the data set
start_data = torch.max(start_data, 0)[1].cuda() if cuda_available else torch.max(start_data, 0)[1] # convert one hot vectors to integers

def prog_callback(step, total_steps):
    print(str(100 * step // total_steps) + "% generated")
start_data.to(device)
generated = model.generate_fast(num_samples=160000,
                                first_samples=start_data,
                                progress_callback=prog_callback,
                                progress_interval=1000,
                                temperature=1.0,
                                regularize=0.)

