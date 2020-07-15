from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from utils.prepare_v02 import signal_regulation, myfft1_norm


class MyModelTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(MyModelTrainer, self).__init__(sess, model, data, config,logger)
        self.min_loss_eval = 100


    def train_epoch(self):
        loop = tqdm(range(self.data.get_epoch_size(self.config.batch_size)))
        losses = []
        # accs = []
        losses_eval = []
        log_it = 0
        for _ in loop:
            loss = self.train_step()
            losses.append(loss)

            log_it += 1
            if (log_it%50 == 49):
                input,decode,loss_b = self.train_step_return_io();
                cur_it = self.model.global_step_tensor.eval(self.sess)
                summaries_dict = {
                    'tr_loss_batch': loss_b,
                    'tr_input_amp': input[:,:,:,2:3],
                    'tr_input_phase': input[:,:,:,3:],
                    'tr_decode_amp':decode[:,:,:,2:3],
                    'tr_decode_phase':decode[:,:,:,3:],
                    'tr_diff': input-decode
                    # 'acc': acc,
                }
                self.logger.summarize(cur_it, summaries_dict=summaries_dict)

                input,decode,loss_b = self.eval_step();
                cur_it = self.model.global_step_tensor.eval(self.sess)
                summaries_dict = {
                    'loss_batch': loss_b,
                    'input_amp': input[:,:,:,2:3],
                    'input_phase': input[:,:,:,3:],
                    'decode_amp':decode[:,:,:,2:3],
                    'decode_phase':decode[:,:,:,3:],
                    'diff': input-decode
                    # 'acc': acc,
                }
                losses_eval.append(loss_b)
                self.logger.summarize(cur_it, summaries_dict=summaries_dict)

            # accs.append(acc)
        loss = np.mean(losses)
        loss_eval = np.mean(losses_eval)
        print(loss)
        summaries_dict = {
            'loss_epoch': loss,
            # 'acc': acc,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        # acc = np.mean(accs)
        if loss_eval< self.min_loss_eval:
            self.model.save(self.sess)
            self.min_loss_eval = loss_eval

    def train_step(self):
        batch_x, batch_y = next(self.data.get_batch_generator(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.is_training: True}
        _, loss = self.sess.run([self.model.train_step, self.model.loss],
                                     feed_dict=feed_dict)

        return loss
    def train_step_return_io(self):
        batch_x, batch_y = next(self.data.get_batch_generator(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.is_training: True}
        _,decode, loss = self.sess.run([self.model.train_step,self.model.decode, self.model.loss],
                                     feed_dict=feed_dict)

        return np.asarray(batch_x), decode,loss

    def eval_step(self):
        batch_x, batch_y = next(self.data.get_batch_generator(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.is_training: False}
        decode, loss = self.sess.run([self.model.decode, self.model.loss],
                                     feed_dict=feed_dict)

        return np.asarray(batch_x), decode,loss
