import logging
import os

import numpy as np
import torch

from federatedscope.core.monitors import Monitor
from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer

logger = logging.getLogger(__name__)


class GraphMiniBatchTrainer(GeneralTorchTrainer):
    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        setattr(ctx, "{}_y_inds".format(ctx.cur_data_split), [])
        if len(ctx.cfg.data.labels_distribution) > 1 and ctx.cfg.criterion.type == "ReweightLoss":
            weight = torch.Tensor(self.ctx.cfg.data.labels_distribution)
            ctx.criterion.set_weight(weight)

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        pred = ctx.model(batch)
        # TODO: deal with the type of data within the dataloader or dataset
        # now we hard code it for cikmcup
        if 'regression' in ctx.cfg.model.task.lower():
            label = batch.y
            # Like [['Normalize', {'mean': [0.1307], 'std': [0.3081]}]]
            if len(ctx.cfg.data.target_transform)==1 and ctx.cfg.data.target_transform[0][0] == "Normalize":
                mean = torch.Tensor(ctx.cfg.data.target_transform[0][1]['mean']).to(ctx.device)
                std = torch.Tensor(ctx.cfg.data.target_transform[0][1]['std']).to(ctx.device)
                origin_label = label.clone().detach()
                true_pred = pred.clone().detach()*std+mean
                label = (label-mean)/std
        else:
            if ctx.cfg.model.out_channels == 1:
                label = batch.y.float()
                pred = torch.sigmoid(pred)
            else:
                label = batch.y.squeeze(-1).long()
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        ctx.loss_batch = ctx.criterion(pred, label)

        ctx.batch_size = len(label)
        if 'regression' in ctx.cfg.model.task.lower() and len(ctx.cfg.data.target_transform)==1 and ctx.cfg.data.target_transform[0][0] == "Normalize":
            ctx.y_true = origin_label
            ctx.y_prob = true_pred
        else:
            ctx.y_true = label
            ctx.y_prob = pred

        # record the index of the ${MODE} samples
        if hasattr(ctx.data_batch, 'data_index'):
            setattr(
                ctx,
                f'{ctx.cur_data_split}_y_inds',
                ctx.get(f'{ctx.cur_data_split}_y_inds') + [batch[_].data_index.item() for _ in range(len(label))]
            )

    def _hook_on_batch_forward_flop_count(self, ctx):
        if not isinstance(self.ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Plz check whether this is you want.")
            return

        if self.cfg.eval.count_flops and self.ctx.monitor.flops_per_sample \
                == 0:
            # calculate the flops_per_sample
            try:
                batch = ctx.data_batch.to(ctx.device)
                from torch_geometric.data import Data
                if isinstance(batch, Data):
                    x, edge_index = batch.x, batch.edge_index
                from fvcore.nn import FlopCountAnalysis
                flops_one_batch = FlopCountAnalysis(ctx.model,
                                                    (x, edge_index)).total()
                if self.model_nums > 1 and ctx.mirrored_models:
                    flops_one_batch *= self.model_nums
                    logger.warning(
                        "the flops_per_batch is multiplied by "
                        "internal model nums as self.mirrored_models=True."
                        "if this is not the case you want, "
                        "please customize the count hook")
                self.ctx.monitor.track_avg_flops(flops_one_batch,
                                                 ctx.batch_size)
            except:
                logger.warning(
                    "current flop count implementation is for general "
                    "NodeFullBatchTrainer case: "
                    "1) the ctx.model takes only batch = ctx.data_batch as "
                    "input."
                    "Please check the forward format or implement your own "
                    "flop_count function")
                self.ctx.monitor.flops_per_sample = -1  # warning at the
                # first failure

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        self.ctx.monitor.total_flops += self.ctx.monitor.flops_per_sample * \
            ctx.batch_size

    def save_prediction(self, path, client_id, task_type):
        y_inds, y_probs = self.ctx.test_y_inds, self.ctx.test_y_prob
        os.makedirs(path, exist_ok=True)

        # TODO: more feasible, for now we hard code it for cikmcup
        if 'regression' in task_type.lower():
            y_preds = y_probs
        else:
            if self.cfg.model.out_channels > 1:
                y_preds = np.argmax(y_probs, axis=-1)
            else:
                y_preds = np.round(y_probs.squeeze(-1)).astype('int')

        if len(y_inds) != len(y_preds):
            raise ValueError(f'The length of the predictions {len(y_preds)} not equal to the samples {len(y_inds)}.')

        with open(os.path.join(path, 'prediction.csv'), 'a') as file:
            for y_ind, y_pred in zip(y_inds,  y_preds):
                if 'classification' in task_type.lower():
                    line = [self.cfg.federate.clients_id[client_id-1], y_ind] + [y_pred]
                else:
                    line = [self.cfg.federate.clients_id[client_id-1], y_ind] + list(y_pred)
                file.write(','.join([str(_) for _ in line]) + '\n')


def call_graph_level_trainer(trainer_type):
    if trainer_type == 'graphminibatch_trainer':
        trainer_builder = GraphMiniBatchTrainer
        return trainer_builder


register_trainer('graphminibatch_trainer', call_graph_level_trainer)
