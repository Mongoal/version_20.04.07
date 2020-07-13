from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from utils.prepare_v02 import signal_regulation

class MyModelTester(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(MyModelTester, self).__init__(sess, model, data, config,logger)

    def train_epoch(self):
        loop = tqdm(range(self.data.get_epoch_size(self.config.batch_size)))
        losses = []
        # accs = []
        for _ in loop:
            loss = self.train_step()
            print(loss)
            losses.append(loss)
            # accs.append(acc)
        loss = np.mean(losses)
        # acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            # 'acc': acc,
        }
        print(loss)
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        batch_x = [signal_regulation(x) for x in batch_x]
        feed_dict = {self.model.x: batch_x }
        _, loss, y, c, u1,u2,u3,u4 = self.sess.run([self.model.train_step, self.model.mse,
                                    self.model.decoding, self.model.end_points['code'],
                                    self.model.end_points['up_conv_1'],self.model.end_points['up_conv_2'],
                                    self.model.end_points['up_conv_3'],self.model.end_points['up_conv_4']],
                                     feed_dict=feed_dict)

        return loss

