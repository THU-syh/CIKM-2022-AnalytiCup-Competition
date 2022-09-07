from federatedscope.core.aggregator import ClientsAvgAggregator


class FinetuneAggregator(ClientsAvgAggregator):
    def __init__(self, model=None, device='cpu', config=None):
        super().__init__(model, device, config)
        self.no_aggregate = False


    def aggregate(self, agg_info):
        """
        To preform aggregation

        Arguments:
        agg_info (dict): the feedbacks from clients
        :returns: the aggregated results
        :rtype: dict
        """
        if not self.no_aggregate:
            models = agg_info["client_feedback"]
            recover_fun = agg_info['recover_fun'] if (
                'recover_fun' in agg_info and self.cfg.federate.use_ss) else None
            avg_model = self._para_weighted_avg(models, recover_fun=recover_fun)
        else:
            avg_model = {}
        return avg_model