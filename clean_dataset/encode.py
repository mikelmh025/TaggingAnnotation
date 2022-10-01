from avatar_stylization.nn_models.encoders.encoder import EncoderModel
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--dummy', type=str, default=None)
opts = parser.parse_args()
opts.stylegan_size = 256
opts.encoder_type = 'GradualVAEStyleEncoder'
opts.input_nc = 3
opts.checkpoint_path = 'clean_dataset/psp_ffhq_encode.pt'
opts.encoder_name = 'psp'

net = EncoderModel(opts)

a=1