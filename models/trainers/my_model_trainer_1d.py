from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from utils.prepare_v02 import signal_regulation

class MyModelTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(MyModelTrainer, self).__init__(sess, model, data, config,logger)

    def train_epoch(self):
        loop = tqdm(range(self.data.get_epoch_size(self.config.batch_size)))
        losses = []
        # accs = []
        log_it = 0
        for _ in loop:
            loss = self.train_step()
            losses.append(loss)

            log_it += 1
            if (log_it%200 == 0):
                input,decode,loss_b = self.eval_step();
                input = input[:,:,0].reshape([input.shape[0],54,60,1])
                decode = decode[:,:,0].reshape([input.shape[0],54,60,1])
                cur_it = self.model.global_step_tensor.eval(self.sess)
                summaries_dict = {
                    'loss_batch': loss_b,
                    'input': input,
                    'decode':decode,
                    'diff': input-decode
                    # 'acc': acc,
                }
                self.logger.summarize(cur_it, summaries_dict=summaries_dict)

            # accs.append(acc)
        loss = np.mean(losses)
        print(loss)
        summaries_dict = {
            'loss_epoch': loss,
            # 'acc': acc,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        # acc = np.mean(accs)

        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        batch_x = [signal_regulation(x) for x in batch_x]
        feed_dict = {self.model.x: batch_x, self.model.is_training: True}
        _, loss = self.sess.run([self.model.train_step, self.model.loss],
                                     feed_dict=feed_dict)

        return loss

    def eval_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        batch_x = [signal_regulation(x) for x in batch_x]
        feed_dict = {self.model.x: batch_x, self.model.is_training: True}
        decode, loss = self.sess.run([self.model.decode, self.model.loss],
                                     feed_dict=feed_dict)

        return np.asarray(batch_x), decode,loss
