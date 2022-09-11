import torch
import os
import logging
import copy
import numpy as np

from federatedscope.core.message import Message
from federatedscope.core.worker.server import Server
from federatedscope.core.worker.client import Client
from federatedscope.core.auxiliaries.utils import merge_dict, merge_param_dict
from federatedscope.core.auxiliaries.utils import param2tensor

logger = logging.getLogger(__name__)

class SmoothClient(Client):
    def _interpolate_model(self, old_model,new_model,model_lambda):
        for key in old_model:
            old_model[key] = param2tensor(old_model[key])*(1-model_lambda)
        for key in new_model:
            new_model[key] = param2tensor(new_model[key])
            old_model[key] += new_model[key] * model_lambda
        return old_model

    def callback_funcs_for_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, \
                                    message.content
        # When clients share the local model, we must set strict=True to
        # ensure all the model params (which might be updated by other
        # clients in the previous local training process) are overwritten
        # and synchronized with the received model
        old_model = copy.deepcopy(self.trainer.ctx.model.state_dict())
        new_model = self._interpolate_model(old_model,content,self._cfg.federate.model_lambda)
        if round <= self._cfg.federate.start_finetune_round:
            self.trainer.update(new_model,
                                strict=self._cfg.federate.share_local_model)
        elif round == self._cfg.federate.start_finetune_round+1:
            self.trainer.ctx.regular_weight = 0
            self.early_stopper.patience = 10
            if getattr(self.trainer.ctx,'local_model',None):
                self.trainer.ctx.local_model.load_state_dict(self.trainer.ctx.best_model,strict=False)
            self.trainer.ctx.model.load_state_dict(self.trainer.ctx.best_model,strict=False)

        self.state = round
        skip_train_isolated_or_global_mode = \
            self.early_stopper.early_stopped and \
            self._cfg.federate.method in ["local", "global", "ditto_finetune"]
        if self.is_unseen_client or skip_train_isolated_or_global_mode:
            # for these cases (1) unseen client (2) isolated_global_mode,
            # we do not local train and upload local model
            sample_size, model_para_all, results = \
                0, self.trainer.get_model_para(), {}
            if skip_train_isolated_or_global_mode:
                logger.info(
                    f"[Local/Global mode] Client #{self.ID} has been "
                    f"early stopped, we will skip the local training")
                self._monitor.local_converged()
        else:
            if self.early_stopper.early_stopped and \
                    self._monitor.local_convergence_round == 0:
                logger.info(
                    f"[Normal FL Mode] Client #{self.ID} has been locally "
                    f"early stopped. "
                    f"The next FL update may result in negative effect")
                self._monitor.local_converged()
            sample_size, model_para_all, results = self.trainer.train()
            if self._cfg.federate.share_local_model and not \
                    self._cfg.federate.online_aggr:
                model_para_all = copy.deepcopy(model_para_all)
            train_log_res = self._monitor.format_eval_res(
                results,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                return_raw=True)
            logger.info(train_log_res)
            if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
                self._monitor.save_formatted_results(train_log_res,
                                                        save_file_name="")

        # Return the feedbacks to the server after local update
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, model_para_all)))


    def callback_funcs_for_evaluate(self, message: Message):
        """
        The handling function for receiving the request of evaluating

        Arguments:
            message: The received message
        """

        # Save prediction result
        if self._cfg.data.type == 'cikmcup' and (message.state+1)%self._cfg.eval.predict_freq==0:
            # Evaluate
            if getattr(self.trainer.ctx,'local_model',None):
                cache_model = copy.deepcopy(self.trainer.ctx.local_model.state_dict())
                self.trainer.ctx.local_model.load_state_dict(self.trainer.ctx.best_model,strict=False)
            else:
                cache_model = copy.deepcopy(self.trainer.ctx.model.state_dict())
                self.trainer.ctx.model.load_state_dict(self.trainer.ctx.best_model,strict=False)
            dir = os.path.join(self._cfg.outdir,f'{message.state}')
            os.makedirs(dir, exist_ok=True)
            self.trainer.save_model(path=os.path.join(dir,f"{self.ID}_bestmodel.pt"))
            self.trainer.evaluate(target_data_split_name='test')
            self.trainer.save_prediction(dir, self.ID, self._cfg.model.task)
            logger.info(f"Client #{self.ID} saved prediction results in {os.path.abspath(os.path.join(dir, 'prediction.csv'))}")
            if getattr(self.trainer.ctx,'local_model',None):
                self.trainer.ctx.local_model.load_state_dict(cache_model,strict=False)
            else:
                self.trainer.ctx.model.load_state_dict(cache_model,strict=False)


        sender = message.sender
        self.state = message.state
        if self.state <= self._cfg.federate.start_finetune_round and message.content is not None:
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)
        if self.early_stopper.early_stopped and self._cfg.federate.method in [
                "local", "global", "ditto_finetune"
        ]:
            metrics = list(self.best_results.values())[0]
        else:
            metrics = {}
            if self._cfg.finetune.before_eval:
                self.trainer.finetune()
            for split in self._cfg.eval.split:
                eval_metrics = self.trainer.evaluate(
                    target_data_split_name=split)

                if self._cfg.federate.mode == 'distributed':
                    logger.info(
                        self._monitor.format_eval_res(eval_metrics,
                                                      rnd=self.state,
                                                      role='Client #{}'.format(
                                                          self.ID),
                                                      return_raw=True))

                metrics.update(**eval_metrics)

            formatted_eval_res = self._monitor.format_eval_res(
                metrics,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                forms='raw',
                return_raw=True)
            logger.info(formatted_eval_res)
            update_best_this_round = self._monitor.update_best_result(
                self.best_results,
                formatted_eval_res['Results_raw'],
                results_type=f"client #{self.ID}",
                round_wise_update_key=self._cfg.eval.
                best_res_update_round_wise_key)
            if update_best_this_round:
                if getattr(self.trainer.ctx,'local_model',None):
                    model_param = self.trainer.ctx.local_model.cpu().state_dict()
                else:
                    model_param = self.trainer.ctx.model.cpu().state_dict()
                self.trainer.ctx.best_model = copy.deepcopy(model_param)
            self.history_results = merge_dict(
                self.history_results, formatted_eval_res['Results_raw'])
            self.early_stopper.track_and_check(self.history_results[
                self._cfg.eval.best_res_update_round_wise_key])

        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=metrics))
