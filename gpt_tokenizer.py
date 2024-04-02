#导入中文训练语料.text
#不使用任何外部库，实现一个简单的分词器Tokenizer，实现encode和decode方法，将中文文本转换为id序列，再转换回文本
#由于是文本比较短，所以不需要考虑分词问题，直接按字分割即可
import os
import jieba
class Tokenizer:
   def __init__(self,config=None):
       self.word_to_id = {}
       self.id_to_word = {}
       if config is not None:
              self.text_dir = config.train_dir
       else:
             self.text_dir = 'data/poem'
       #如果有词表，直接读取词表
       if os.path.exists('data/vocab.txt'):
              with open('data/vocab.txt', 'r', encoding='utf-8') as f:
                #特殊字符
                special_words = [' ', '\n', '\t','\u3000','\r','\xa0']
                for i, word in enumerate(special_words):
                    self.word_to_id[word] = i
                    self.id_to_word[i] = word
                for i, word in enumerate(f):
                    word = word.strip()
                    self.word_to_id[word] = i + len(special_words)
                    self.id_to_word[i + len(special_words)] = word

       else:
            #如果是文件夹，读取文件夹下所有文件
            if os.path.isdir(self.text_dir):
                text = ''
                for file in os.listdir(self.text_dir):
                    with open(os.path.join(self.text_dir, file), 'r', encoding='utf-8') as f:
                        text += f.read()
                self.text_path = 'data/text.txt'
                with open(self.text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
            else:
                self.text_path = self.text_dir
            with open(self.text_path, 'r', encoding='utf-8') as f:
                text = f.read()
                vocab = set(text)
                for i, word in enumerate(vocab):
                    self.word_to_id[word] = i
                    self.id_to_word[i] = word
       #保存一下词表
            with open('data/vocab.txt', 'w', encoding='utf-8') as f:
                for word in self.word_to_id:
                    f.write(word + '\n')


   def encode(self, text):
        return [self.word_to_id[word] for word in text]
   def decode(self, ids):
        return ''.join([self.id_to_word[id] for id in ids])
   def get_vocab_size(self):
        return len(self.word_to_id)







if __name__ == '__main__':
    tokenizer = Tokenizer()
    words='没关系的，跟Sakura在外面到处玩，很开心，所以我能坚持下来。'
    ids = tokenizer.encode(words)
    print(ids)
    print(tokenizer.decode(ids))
