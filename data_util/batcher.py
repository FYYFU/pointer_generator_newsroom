from . import batch, data
from .example import Example

import queue as Queue
import time
from random import shuffle
from threading import Thread
import  logging

import random
random.seed(1234)


class Batcher(object):

    BATCH_QUEUE_MAX = 1000
    logging.basicConfig(filename='./log.txt', level=logging.DEBUG)

    def __init__(self, data_path, vocab, mode, batch_size, single_pass):
        self._data_path = data_path
        self._vocab = vocab
        self._single_pass = single_pass
        self.mode = mode
        self.batch_size = batch_size

        self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX)

        if single_pass:
            self._num_example_q_threads = 1
            self._num_batch_q_threads = 1
            self._bucketing_cache_size = 1
            self._finished_reading = False
        else:
            self._num_example_q_threads = 4
            self._num_batch_q_threads = 4
            self._bucketing_cache_size = 8

        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()

        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        if not single_pass:
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.damon = True
            self._watch_thread.start()

    def next_batch(self):

        if self._batch_queue.qsize() == 0:
            logging.info('Bucket input queue is empty when calling next_batch. '
                         'Bucket queue size: %i, Input queue size: %i',
                         self._batch_queue.qsize(), self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                logging.info('Finished reading dataset in single_pass mode')
                return None

        batch = self._batch_queue.get()
        return batch

    def fill_example_queue(self):
        input_gen = data.example_generator(self._data_path)

        while True:
            try:
                (title, summary, article) = next(input_gen)
            except StopIteration:
                logging.info("The example generator for this example_queue"
                             "filling thread has exhausted data")
                if self._single_pass:
                    logging.info("single_pass mode is on, so we've finished reading"
                                 "dataset, This thread is stopping!")
                    self._finished_reading = True
                    break
                else:
                    raise Exception("Single_pass mode is off but the example generator is"
                                    "out of data: error")

            summary_sentences = [sent.strip() for sent in
                                    data.summary2sents(summary)]
            article_sentences = [sent.strip() for sent in
                                    data.summary2sents(article)]

            # 这一个部分本来是没有的，但是为了尝试缩小文章的体积从而减少显存占用。
            # article_sentences = article_sentences[:6]

            example = Example(article_sentences, summary_sentences, self._vocab)
            self._example_queue.put(example)

    def fill_batch_queue(self):
        while True:
            if self.mode == 'decode':
                # 使用一个example在decoder的时候进行beam search
                ex = self._example_queue.get()
                b = [ex for _ in range(self.batch_size)]
                self._batch_queue.put(batch.Batch(b, self._vocab, self.batch_size))
            else:
                inputs = []
                for _ in range(self.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key= lambda input: input.enc_len, reverse=True)

                batches = []
                for i in range(0, len(inputs), self.batch_size):
                    batches.append(inputs[i: i + self.batch_size])

                if not self._single_pass:
                    shuffle(batches)

                for b in batches:
                    self._batch_queue.put(batch.Batch(b, self._vocab, self.batch_size))

    def watch_threads(self):

        while True:
            time.sleep(60)

            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():
                    # example queue 的线程dead，需要restart
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():
                    # batch queue 的线程dead，需要restart
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()





