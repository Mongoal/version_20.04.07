from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from utils.prepare_v03 import signal_regulation,myfft2,myfft1



class MyModelTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(MyModelTrainer, self).__init__(sess, model, data, config,logger)
        self.min_loss_eval = 100
        if config.get('stft_method') == 'myfft1':
            self.myfft = myfft1
            print("use myfft1")
        else:
            self.myfft = myfft2

    def train_epoch(self):
        loop = tqdm(range(self.data.get_epoch_size(self.config.batch_size)))
        losses = []
        accs = []
        losses_eval = []
        accs_eval=[]
        for i in loop:
            loss,acc = self.train_step()
            loss_eval, acc_eval = self.eval_step();
            losses.append(loss)
            accs.append(acc)
            losses_eval.append(loss_eval)
            accs_eval.append(acc_eval)
            if i % 20 == 0:
                summaries_dict = {
                    'eval_acc_batch': acc_eval,
                }
                cur_it = self.model.global_step_tensor.eval(self.sess)
                self.logger.summarize(cur_it, summaries_dict=summaries_dict)

        loss = np.mean(losses)
        loss_eval = np.mean(losses_eval)
        acc = np.mean(accs)
        acc_eval = np.mean(accs_eval)
        summaries_dict = {
            'loss_epoch': loss,
            'acc_epoch': acc,
            'eval_loss_epoch': loss_eval,
            'eval_acc_epoch': acc_eval,
        }
        cur_it = self.model.global_step_tensor.eval(self.sess)
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        # acc = np.mean(accs)
        if loss_eval< self.min_loss_eval:
            self.model.save(self.sess)
            self.min_loss_eval = loss_eval

    def train_step(self):
        batch_x, batch_y = next(self.data.get_train_batch_generator(self.config.batch_size))
        if self.config.h5_data_key=="signals":
            batch_x = [origin_signal[:, 0] + np.asarray(1j, np.complex64) * origin_signal[:, 1] for origin_signal in batch_x]
            batch_x =[ self.myfft(x -np.mean(x,axis=1,keepdims=True),*self.config.stft_args) for x in batch_x]

        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss,acc = self.sess.run([self.model.train_op, self.model.loss,self.model.acc],
                                     feed_dict=feed_dict)

        return loss,acc


    def eval_step(self):
        batch_x, batch_y = next(self.data.get_test_batch_generator(self.config.batch_size))
        if self.config.h5_data_key=="signals":
            batch_x = [origin_signal[:, 0] + np.asarray(1j, np.complex64) * origin_signal[:, 1] for origin_signal in batch_x]
            batch_x =[ self.myfft(x -np.mean(x,axis=1,keepdims=True),*self.config.stft_args) for x in batch_x]

        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: False}
        loss,acc = self.sess.run([self.model.loss,self.model.acc],
                                     feed_dict=feed_dict)

        return loss,acc
