from flgo.experiment.logger.simple_logger import SimpleLogger
class FedBalanceLogger(SimpleLogger):
    """This logger only records metrics on global testing dataset"""

    def get_output_name(self):
        if not hasattr(self, 'option'): raise NotImplementedError('logger has no attr named "option"')
        return "{}_epochs_{}_aggnum_{}_fs_index_{}_hl_index_{}.json".format(self.option["algorithm"],
                                                                            self.option["num_epochs"],
                                                                            self.option["agg_num"],
                                                                            self.option["fs_index"],
                                                                            self.option["hl_index"])
