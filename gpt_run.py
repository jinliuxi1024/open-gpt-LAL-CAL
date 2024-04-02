from gpt_model import GPTmodel
from gpt_tokenizer import Tokenizer
from gpt_train import Trainer
from gpt_utils import Config
import torch

def get_default_config():
    config = Config()
    config.batch_size = 32
    config.block_size = 64
    config.embed_size = 256
    config.is_need = 'train'
    config.train_dir = 'data/poem'
    config.lr = 3e-4
    config.epochs = 2000
    config.device = torch.device('mps')
    config.head_size = 16
    config.layer_size = 8
    config.model = GPTmodel(config).to('mps')
    config.vocab_size = Tokenizer(config).get_vocab_size()
    return config




def train_and_test():
    config = get_default_config()
    trainer = Trainer(config)
    trainer.run(config)

def load_and_test():
    config = get_default_config()
    model = GPTmodel(config).to('mps')
    model.load_state_dict(torch.load('model.pth', map_location='mps'))
    word = '灯火阑珊处'
    idx = torch.tensor([Tokenizer().encode(word)], dtype=torch.long).to('mps')
    idx = model.generate(idx, max_new_len=100)


def load_and_train():
    config = get_default_config()
    model = GPTmodel(config).to('mps')
    model.load_state_dict(torch.load('model.pth', map_location='mps'))
    config.model = model
    trainer = Trainer(config)
    trainer.run(config)

if __name__ == '__main__':
    #train_and_test()
    #load_and_train()
    load_and_test()