from flgo.experiment.logger.simple_logger import SimpleLogger
class FedBalanceLogger(SimpleLogger):
    """This logger only records metrics on global testing dataset"""

    def get_output_name(self,suffix='.json'):
        if not hasattr(self, 'option'): raise NotImplementedError('logger has no attr named "option"')
        output_name= "{}_{}_{}b_{}t".format(self.option["model"],self.option["algorithm"],self.option["b"],self.option["t"])
        return output_name+suffix
    def log_once(self, *args, **kwargs):
        test_metric = self.server.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        self.show_current_output()
