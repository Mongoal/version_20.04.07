from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from utils.prepare_v02 import signal_regulation, myfft1_norm


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
            if (log_it%50 == 49):
                # is_training = True
                input,decode,kl_loss_b,recon_loss_b = self.train_step_return_io();
                cur_it = self.model.global_step_tensor.eval(self.sess)
                summaries_dict = {
                    'tr_kl_loss_batch': kl_loss_b,
                    'tr_recon_loss_batch': recon_loss_b,
                    'tr_input_amp': input[:,:,:,2:3],
                    'tr_decode_amp':decode[:,:,:,2:3],
                    'tr_diff': input-decode
                    # 'acc': acc,
                }
                self.logger.summarize(cur_it, summaries_dict=summaries_dict)
                # is_training = False
                input,decode,kl_loss_b,recon_loss_b = self.eval_step();
                cur_it = self.model.global_step_tensor.eval(self.sess)
                summaries_dict = {
                    'kl_loss_batch': kl_loss_b,
                    'recon_loss_batch': recon_loss_b,
                    'input_amp': input[:, :, :, 2:3],
                    'decode_amp': decode[:, :, :, 2:3],
                    'diff': input - decode
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
        batch_x, batch_y = next(self.data.get_batch_generator(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.is_training: True}
        _, loss = self.sess.run([self.model.train_step, self.model.loss],
                                     feed_dict=feed_dict)

        return loss
    def train_step_return_io(self):
        batch_x, batch_y = next(self.data.get_batch_generator(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.is_training: True}
        _, decode, kl_loss, recon_loss = self.sess.run([self.model.train_step,
                                        self.model.decode,
                                        self.model.kl_loss,
                                        self.model.recon_loss],
                                     feed_dict=feed_dict)

        return np.asarray(batch_x), decode, kl_loss, recon_loss

    def eval_step(self):
        batch_x, batch_y = next(self.data.get_batch_generator(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.is_training: False}
        decode, kl_loss, recon_loss = self.sess.run([self.model.decode,
                                                    self.model.kl_loss,
                                                    self.model.recon_loss],
                                     feed_dict=feed_dict)

        return np.asarray(batch_x), decode, kl_loss, recon_loss
