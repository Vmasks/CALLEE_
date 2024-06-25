import pickle
import logging
from gensim.models import Word2Vec
from time import time
from typing import Union
import os
logging.basicConfig(level=logging.DEBUG)

class GraphEmbedding(object):
    def __init__(self, load_seq: Union[str, list], save_model: str, vector_size: int, window: int, epochs: int) -> None:
        self.load_seq = load_seq
        self.save_model = save_model
        self.vector_size = vector_size
        self.window = window
        self.epochs = epochs
        self.read_data()


    def read_data(self):
        logging.info('正在读取序列')
        if isinstance(self.load_seq, str):
            with open(self.load_seq, 'rb') as f:
                self.seq = pickle.load(f)
        elif isinstance(self.load_seq, list):
            self.seq = self.load_seq
        else:
            raise TypeError('load_seq type error.')
        logging.info('序列加载完成')
            
        
    def embedding(self):
        self.model = Word2Vec(vector_size=self.vector_size,
                            window=self.window,
                            sg=1, #Skip-Gram
                            hs=0,
                            negative=10,
                            alpha=0.03,
                            min_alpha=0.0007,
                            workers=10
                        )
        t = time()
        self.model.build_vocab(self.seq, progress_per=100000)
        logging.info('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
        # train
        t = time()
        self.model.train(self.seq, total_examples=self.model.corpus_count, epochs=self.epochs, report_delay=1)
        logging.info('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
        
        if self.save_model:
            self.model.save(os.path.join(self.save_model, f'{self.vector_size}_{self.window}_{self.epochs}.model'))
        
        return self.model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description = "graph_embedding: input seq and output node embedding"
    )
    parser.add_argument('--load_seq', help='path of input seq file')
    parser.add_argument('--save_model', help='path of output model file')
    parser.add_argument('--vector_size', default=32, type=int, help='embedding vector dimensions')
    parser.add_argument('--window_size', default=5, type=int, help='size of word2vec window')
    parser.add_argument('--epochs', default=20, type=int, help='word2vec epochs')
    args = parser.parse_args()

    ge = GraphEmbedding(args.load_seq, args.save_model, args.vector_size, args.window_size, args.epochs)
    ge.embedding()