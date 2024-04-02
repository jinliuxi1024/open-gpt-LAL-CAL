from gpt_tokenizer import Tokenizer
#测试一下数据集生成\
from gpt_model import GPTmodel
from gpt_train import Trainer
import torch
class Config:
    def __init__(self):
        self.batch_size = 32
        self.block_size = 64
        self.embed_size = 256
        self.is_need = 'train'
        self.vocab_size =Tokenizer().get_vocab_size()
        self.lr =3e-4
        self.epochs = 2000
        self.device = torch.device('mps')
        self.head_size = 16
        self.layer_size = 8
        self.model = GPTmodel(self).to('mps')
        self.train_dir = 'data/poem'

config = Config()
def test_tokenizer():
    tokenizer = Tokenizer()
    words='没关系的，跟Sakura在外面到处玩，很开心，所以我能坚持下来。'
    ids = tokenizer.encode(words)
    print(ids)
    words='没关系的，跟Sakura在外面到处玩，很开心，所以我能坚持下来。\n'
    print(ids)
    print(tokenizer.decode(ids))
def test_trainer():
    trainer = Trainer(config)
    train_x, train_y = trainer.get_data(config)
    print(train_x)
    print(train_y)
def test_embedding():
    embedding = torch.nn.Embedding(config.vocab_size, config.embed_size)
    x = torch.randint(0, config.vocab_size, (config.batch_size, config.block_size))
    print(x.shape)
    print(embedding(x).shape)

if __name__ == '__main__':
    test_tokenizer()
    test_trainer()
    test_embedding()
    print('done')



