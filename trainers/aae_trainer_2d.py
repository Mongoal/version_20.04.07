from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from utils.prepare_v02 import signal_regulation, myfft1_norm


class MyModelTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(MyModelTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.data.get_epoch_size(self.config.batch_size)))
        losses = []
        # accs = []
        log_it = 0
        for _ in loop:
            loss = self.train_step()
            losses.append(loss)

            log_it += 1
            if (log_it % 50 == 49):
                # is_training = True
                input, decode, recon_loss_curr, D_loss_curr, G_loss_curr = self.train_step_return_io();
                cur_it = self.model.global_step_tensor.eval(self.sess)
                summaries_dict = {
                    'tr_recon_loss_curr': recon_loss_curr,
                    'tr_D_loss_curr': D_loss_curr,
                    'tr_G_loss_curr': G_loss_curr,
                    'tr_input_amp': input[:, :, :, 2:3],
                    'tr_decode_amp': decode[:, :, :, 2:3],
                    'tr_diff': input - decode
                    # 'acc': acc,
                }
                self.logger.summarize(cur_it, summaries_dict=summaries_dict)
                # is_training = False
                input, decode, recon_loss_curr, D_loss_curr, G_loss_curr = self.eval_step();
                cur_it = self.model.global_step_tensor.eval(self.sess)
                summaries_dict = {
                    'recon_loss_curr': recon_loss_curr,
                    'D_loss_curr': D_loss_curr,
                    'G_loss_curr': G_loss_curr,
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
            'recon_loss_curr_epoch': loss,
            # 'acc': acc,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        # acc = np.mean(accs)

        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.get_batch_generator(self.config.batch_size))

        _, recon_loss_curr = self.sess.run([self.model.AE_solver, self.model.recon_loss],
                                           feed_dict={self.model.x: batch_x, self.model.is_training: True})
        _, D_loss_curr = self.sess.run([self.model.D_solver, self.model.D_loss],
                                       feed_dict={self.model.x: batch_x,
                                                  self.model.is_training: True, 
                                                  self.model.z: np.random.randn(len(batch_x), self.model.Z_DIM)})
        _, G_loss_curr = self.sess.run([self.model.G_solver, self.model.G_loss],
                                       feed_dict={self.model.x: batch_x,
                                                  self.model.z: np.random.randn(len(batch_x), self.model.Z_DIM),
                                                  self.model.is_training: True})

        return recon_loss_curr

    def train_step_return_io(self):

        batch_x, batch_y = next(self.data.get_batch_generator(self.config.batch_size))

        feed_dict = {self.model.x: batch_x,
                     self.model.is_training: True,
                     self.model.z: np.random.randn(len(batch_x), self.model.Z_DIM)}

        _, decode, recon_loss_curr, D_loss_curr, G_loss_curr = self.sess.run(
            [self.model.AE_solver, self.model.decode, self.model.recon_loss, self.model.D_loss, self.model.G_loss],
            feed_dict=feed_dict)

        return np.asarray(batch_x), decode, recon_loss_curr, D_loss_curr, G_loss_curr

    def eval_step(self):

        batch_x, batch_y = next(self.data.get_batch_generator(self.config.batch_size))

        feed_dict = {self.model.x: batch_x,
                     self.model.is_training: False,
                     self.model.z: np.random.randn(len(batch_x), self.model.Z_DIM)}

        decode, recon_loss_curr, D_loss_curr, G_loss_curr = self.sess.run(
            [ self.model.decode, self.model.recon_loss, self.model.D_loss, self.model.G_loss],
            feed_dict=feed_dict)

        return np.asarray(batch_x), decode, recon_loss_curr, D_loss_curr, G_loss_curr
