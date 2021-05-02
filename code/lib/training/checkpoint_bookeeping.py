import os

import torch
import numpy as np

class CheckpointUpdate:
    """
    """
    def __init__(self, net, first_iter_save, out_dir, sha, nbest=None):

        self.net = net
        self.first_iter_save = first_iter_save
        # self.iter_save = iter_save
        self.directory = out_dir + f'/checkpoint_{sha}'
        self.nbest = nbest
        self.best_scores = []

    def update(self, iteration, epoch, score=None, max=True):
        if self.nbest is None:
            self._periodic_save(iteration, epoch)
        else:
            self._best_score_save(iteration, epoch, score, max)

    def _periodic_save(self, iteration, epoch):
        if iteration > self.first_iter_save:
            torch.save({
                'state_dict': self.net.state_dict(),
                'iteration': iteration,
                'epoch': epoch,
            }, self.directory + f'/%08d_model.pth' % iteration)

    def _best_score_save(self, iteration, epoch, score, max):

        # if iteration not in self.iter_save: return
        if iteration <= self.first_iter_save: return

        _name = f"{iteration:08}_{score:.6f}_model.pth"

        # print('\n', self.best_scores, '\n')

        save = False
        if len(self.best_scores) < self.nbest:
            self.best_scores.append([score, _name, iteration])
            save = True
        else:
            _tmp = np.array(self.best_scores)
            min_score = _tmp[:, 0].astype(float).min()
            min_index = _tmp[:, 0].astype(float).argmin()
            max_score = _tmp[:, 0].astype(float).max()
            max_index = _tmp[:, 0].astype(float).argmax()

            if score > min_score and max:
                _, model_to_remove, _ = self.best_scores.pop(min_index)
                os.remove(os.path.join(self.directory, model_to_remove))
                self.best_scores.append([score, _name, iteration])
                save = True
            elif score > max_score and not max:
                _, model_to_remove, _ = self.best_scores.pop(max_index)
                os.remove(os.path.join(self.directory, model_to_remove))
                self.best_scores.append([score, _name, iteration])
                save = True

        # print('\n save=' , save, '\n')

        if save:
            torch.save({
                'state_dict': self.net.state_dict(),
                'iteration': iteration,
                'epoch': epoch,
            }, os.path.join(self.directory, _name))
