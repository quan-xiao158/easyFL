import torch
from flgo.utils import fmodule
from flgo.algorithm.asyncbase import AsyncServer
from flgo.algorithm.fedbase import BasicServer, BasicClient
from collections import deque
import copy
from tqdm import tqdm


class Server(AsyncServer):
    def __init__(self, option={}):
        super(Server, self).__init__(option)
        self.server_send_round = [0] * 100  # 服务器下发模型round记录
        self.client_fs_list = [0] * 100  #服务器记录客户端时间差列表
        self.client_lh_list = [999] * 100  #服务器记录客户端上传模型陈旧度列表
        self.commit_num = [0] * 100
        self.fl_queue = deque()
        self.fh_queue = deque()
        self.sl_queue = deque()
        self.sh_queue = deque()
        self.round_number = 0
        self.maxlambda = 0
        self.total_traffic = 0
        self.additional_traffic = 0
        self.lambdalist = []
        '''
        fedbalance  超参数：
        
        '''
        self.alpha = self.option['alpha']
        self.agg_num = self.option['agg_num']
        self.fs_index = self.option['fs_index']
        self.hl_index = self.option['hl_index']

    def package_handler(self, received_packages: dict):
        """
        Handle packages received from clients and return whether the global model is updated in this function.

        Args:
            received_packages (dict): a dict consisting of uploaded contents from clients
        Returns:
            is_model_updated (bool): True if the global model is updated in this function.
        """
        if self.is_package_empty(received_packages): return False
        self.model = self.aggregate(received_packages['model'])
        return True

    def is_package_empty(self, received_packages: dict):
        """
        Check whether the package dict is empty

        Returns:
            is_empty (bool): True if the package dict is empty
        """
        return len(received_packages['__cid']) == 0

    def iterate(self):
        """
        1、sample()采样出两个客户端，模拟异步下有两个客户端提交模型，得到客户端编号列表，selected_clients
        2、communicate（）与客户端通信，得到客户端上的两个模型，received_packages
        4、package_handler（）对模型进行聚合（并下发）
        5、返回全局模型是否更新
        """
        self.selected_client = self.sample_async()

        self.concurrent_clients.add(self.selected_client)
        self.commit_num[self.selected_client] += 1
        #如果有客户端被选中，记录一条日志，显示被选中的客户端及当前时间
        self.model._round = self.current_round  #将模型的 _round 属性设置为当前的回合数

        fs_tag = self.compute_fs(self.selected_client)
        selected_clients = [self.selected_client]
        received_packages = self.communicate(selected_clients, None, 2)
        client_model = received_packages['model']
        client_lambda = self.compute_lambda(client_model)
        lh_tag = self.compute_hl(self.selected_client, client_lambda)
        self.current_round = self.current_round + 1
        #xiafamox
        return self.core_select_algorithm(self.selected_client, fs_tag, lh_tag, client_model)

    def run(self):
        """
        Running the FL symtem where the global model is trained and evaluated iteratively.
        """
        self.gv.logger.time_start('Total Time Cost')
        if not self._load_checkpoint() and self.eval_interval > 0:
            # evaluating initial model performance
            self.gv.logger.info("--------------Initial Evaluation--------------")
            self.gv.logger.time_start('Eval Time Cost')
            self.gv.logger.log_once()  # 初始评估全局模型的准确度
            self.gv.logger.time_end('Eval Time Cost')
        all_clients = self.available_clients if 'available' in self.sample_option else [cid for cid in
                                                                                        range(self.num_clients)]
        all_clients = list(set(all_clients).difference(self.buffered_clients))
        #todo 高陈旧度模型与全局模型聚合一次后发给快客户端训练
        self.gv.logger.info("-------------- 执行fedavg--------------")
        favmodel = self.fedavg_agg(all_clients)
        self.model = favmodel
        self.gv.logger.info("--------------Initial Evaluation--------------")
        self.gv.logger.time_start('Eval Time Cost')
        self.gv.logger.log_once()  # 初始评估全局模型的准确度
        self.gv.logger.time_end('Eval Time Cost')
        self.communicate(all_clients, self.model, 1)  #下发所有模型到客户端上
        self.gv.logger.info("--------------全局模型下发--------------")
        while True:
            if self.round_number > self.num_rounds: break
            self.gv.clock.step()
            # iterate
            updated = self.iterate()  # 进行客户端选择迭代训练和模型聚合
            # using logger to evaluate the model if the model is updated
            if updated is True:
                self.gv.logger.info("--------------Round {}--------------".format(self.round_number))
                # check log interval
                if self.gv.logger.check_if_log(self.current_round, self.eval_interval):
                    self.gv.logger.time_start('Eval Time Cost')
                    self.gv.logger.log_once()  # 验证模型损失和准确率
                    self.gv.logger.time_end('Eval Time Cost')
                    self._save_checkpoint()
                # check if early stopping
                if self.gv.logger.early_stop(): break
                self.current_round += 1
                # decay learning rate
                self.global_lr_scheduler(self.current_round)
                if self.round_number % 50 == 0:
                    self.gv.logger.save_output_as_json()
        self.gv.logger.info("=================End==================")
        self.gv.logger.time_end('Total Time Cost')
        # save results as .json file
        self.gv.logger.save_output_as_json()
        print("总通信量{}".format(self.total_traffic))
        print("额外通信量{}".format(self.additional_traffic))
        print("占比{}".format(self.total_traffic/self.total_traffic))
        return

    def median(self, lst):
        import copy
        dp = copy.deepcopy(lst)
        dp.sort()
        return dp[self.fs_index]

    def compute_lambda(self, w_local):
        w_global = self.model
        l2_norm_squared = self.compute_l2_difference(w_global, w_local)

        return abs(l2_norm_squared)

    def compute_hl(self, client_id, lambda_i):
        self.lambdalist.append(lambda_i)
        self.client_lh_list[client_id] = lambda_i
        # filtered_list = list(filter(lambda x: x != -1, self.client_lh_list))
        # # 计算均值
        # mean_value = sum(filtered_list) / len(filtered_list) if filtered_list else 0
        copy_lh_list = copy.deepcopy(self.client_lh_list)
        copy_lh_list.sort()
        if lambda_i <= copy_lh_list[self.hl_index]:
            return "L"
        else:
            return "H"

    def compute_fs(self, client_id):
        for id in range(100):
            self.client_fs_list[id] = self.current_round - self.server_send_round[id]

        time_delay = self.current_round - self.server_send_round[client_id]
        self.server_send_round[self.selected_client] = self.current_round
        self.client_fs_list[client_id] = time_delay

        media = self.median(self.client_fs_list)

        if time_delay <= media:
            return "F"
        else:
            return "S"

    def core_select_algorithm(self, client_id, fs_tag, lh_tag, model):
        agg_num = self.agg_num
        if lh_tag == "H" and fs_tag == "S":
            if len(self.fl_queue) != 0:
                th = self.fl_queue.pop()
                client2_id = th["client_id"]
                client2_model = th["model"]
                self.communicate([client_id], client2_model, 1)
                self.communicate([client2_id], model, 1)
                self.additional_traffic += 4
                self.concurrent_clients.difference_update([client_id, client2_id])
                return False
            else:
                self.sh_queue.append({"client_id": client_id, "model": model})
                return False
        elif lh_tag == "H" and fs_tag == "F":
            if len(self.fh_queue) == agg_num:
                self.fh_queue.append({"client_id": client_id, "model": model})
                id_list, model_list = self.queue_pop([self.fh_queue])
                agg_model = self.fedbalance_late_aggregate(model_list, id_list)
                self.communicate(id_list, agg_model, 1)
                self.additional_traffic+=len(id_list)*2
                self.concurrent_clients.difference_update(id_list)
                self.round_number += 1
                return True
            else:
                self.fh_queue.append({"client_id": client_id, "model": model})
                return False
        elif lh_tag == "L" and fs_tag == "S":
            #1、将模型入队，如果模型数量长度
            if len(self.sl_queue) + len(self.fl_queue) == agg_num:
                self.sl_queue.append({"client_id": client_id, "model": model})
                id_list, model_list = self.queue_pop([self.sl_queue, self.fl_queue])
                agg_model = self.fedbalance_late_aggregate(model_list, id_list)
                self.model = agg_model
                self.communicate(id_list, agg_model, 1)
                self.concurrent_clients.difference_update(id_list)
                self.round_number += 1
                return True
            else:
                self.sl_queue.append({"client_id": client_id, "model": model})
                return False
        else:  #LF
            if len(self.sh_queue) != 0:
                th = self.sh_queue.pop()
                client2_id = th["client_id"]
                client2_model = th["model"]
                self.communicate([client_id], client2_model, 1)
                self.communicate([client2_id], model, 1)
                self.concurrent_clients.difference_update([client_id, client2_id])
                return False
            # 1、将模型入队，如果模型数量长度
            elif len(self.sl_queue) + len(self.fl_queue) == agg_num:
                self.fl_queue.append({"client_id": client_id, "model": model})
                id_list, model_list = self.queue_pop([self.sl_queue, self.fl_queue])
                agg_model = self.fedbalance_late_aggregate(model_list, id_list)
                self.model = agg_model
                self.communicate(id_list, agg_model, 1)
                self.concurrent_clients.difference_update(id_list)
                return False
            else:
                self.fl_queue.append({"client_id": client_id, "model": model})
                return False

    def fedbalance_late_aggregate(self, models, client_ids):
        """
        聚合来自不同客户端的模型，考虑模型更新的延迟。

        参数:
        - models (list): 模型列表，每个元素代表一个客户端的模型。
        - client_ids (list): 客户端ID列表，对应于models中的每个模型。
        - late_list (list): 延迟列表，对应于client_ids，每个元素表示该客户端模型的延迟（current_round - tau）。

        返回:
        - bool: 如果聚合成功，返回True；如果输入为空或不合法，返回False。
        """
        # dl = []
        # for id in self.selected_clients:
        #     dl.append(self.round_number - self.server_send_round[id])
        #     self.server_send_round[id] = self.round_number
        #
        # alpha_ts = [self.alpha * self.s(late) for late in dl]

        # commmit_avg=sum(self.commit_num)/100
        # commit_num_list=[]
        # for id in client_ids:
        #     commit_num_list.append(abs(self.commit_num[id]-commmit_avg))

        # alpha_ts = [self.alpha * self.s(late) for late in commit_num_list]
        alpha_ts = [0.6] * 16
        # 4. 更新每个接收到的模型，与当前模型进行加权融合

        currently_updated_models = []
        for alpha_t, model_k in zip(alpha_ts, models):
            # 假设 self.model 和 model_k 支持标量乘法和加法操作
            updated_model = (1 - alpha_t) * self.model + alpha_t * model_k
            currently_updated_models.append(updated_model)

        # 5. 聚合所有更新后的模型，更新当前模型

        return self.fedbalance_aggregate(currently_updated_models, client_ids)

    def help_agg(self, model, cid):
        pass

    def s(self, delta_tau):
        return (delta_tau + 1) ** (-0.5)

    def fedbalance_aggregate(self, models, client_ids):
        """
        聚合多个客户端的模型，并将结果与 self.model 进行加权聚合。

        参数：
            models (list): 包含多个客户端模型参数的列表（PyTorch tensors）。
            client_ids (list): 包含多个客户端标识符的列表。

        返回：
            dict: 聚合后的最终模型。
        """
        # 计算所有客户端的数据量总和

        total_data_vol = sum([self.clients[client_id].datavol for client_id in client_ids])

        # 计算每个客户端模型的权重
        weights = [self.clients[client_id].datavol / total_data_vol for client_id in client_ids]

        # 加权聚合所有模型
        weighted_models = [weight * model for weight, model in zip(weights, models)]
        aggregated_model = fmodule._model_sum(weighted_models)

        return aggregated_model

    def communicate(self, client_id_list, send_model, mtype):
        self.total_traffic += len(client_id_list)
        for client_id in client_id_list:
            server_pkg = self.fedbalance_pack_model(send_model)  #全局模型
            server_pkg['__mtype__'] = mtype
            client_model = self.communicate_with(client_id, package=server_pkg)  # 与客户端进行通信，下发全局模型到本机训练，获取客户端上的模型
            if mtype == 2 or mtype == 0:
                return client_model

    def fedbalance_pack_model(self, send_model):
        return {
            "model": copy.deepcopy(send_model),
        }

    def compute_l2_difference(self, model1, model2):
        # 获取模型的所有参数
        params1 = [p.data for p in model1.parameters()]
        params2 = [p.data for p in model2.parameters()]

        # 用来存储总的 L2 范数差异
        total_l2_diff = 0.0

        # 对每一层的权重和偏置计算 L2 范数差异
        for p1, p2 in zip(params1, params2):
            # 计算 L2 范数差异并加到总差异中
            l2_diff = torch.norm(p1 - p2, p=2)  # L2范数
            total_l2_diff += l2_diff.item() ** 2  # L2范数的平方和

        # 取平方根得到最终的 L2 范数差异
        total_l2_diff = torch.sqrt(torch.tensor(total_l2_diff))
        return total_l2_diff

    def queue_pop(self, queue_list):
        id_list = []
        model_list = []
        for q in queue_list:
            while len(q) != 0:
                th = q.pop()
                id_list.append(th['client_id'])
                model_list.append(th['model'])

        return id_list, model_list

    def fedavg_agg(self, all_clients):
        model_list = []
        communicate_clients = list(set(all_clients))  # 去重后的客户端列表，确保每个客户端只通信一次
        for client_id in tqdm(communicate_clients,
                              desc="Local Training on {} Clients".format(len(communicate_clients)), leave=False):
            server_pkg = self.fedbalance_pack_model(self.model)  #全局模型
            server_pkg['__mtype__'] = 0
            client_model = self.communicate_with(client_id, package=server_pkg)  # 与客户端进行通信，下发全局模型到本机训练，获取客户端上的模型
            model_list.append(client_model['model'])
        return self.fedavg_aggregate(model_list, all_clients)

    def fedavg_aggregate(self, models, all_clients):
        local_data_vols = [c.datavol for c in self.clients]
        total_data_vol = sum(local_data_vols)
        p = [1.0 * local_data_vols[cid] / total_data_vol for cid in all_clients]
        sump = sum(p)
        p = [pk / sump for pk in p]
        return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])


class Client(BasicClient):
    def __init__(self, option={}):
        super().__init__(option)
        self.mu = None

    def initialize(self, *args, **kwargs):
        self.actions = {0: self.reply, 1: self.send_model, 2: self.rp}
        self.mu = 0.1

    def send_model(self, servermodel):
        model = self.unpack(servermodel)
        self.model = model

    def rp(self, servermodel):
        self.model_train(self.model)
        cpkg = self.pack(self.model)
        return cpkg

    @fmodule.with_multi_gpus
    def model_train(self, model):
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        for iter in range(10):
            # get a batch of data
            batch_data = self.get_batch_data()
            self.model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss_proximal = 0
            for pm, ps in zip(self.model.parameters(), src_model.parameters()):
                loss_proximal += torch.sum(torch.pow(pm - ps, 2))
            loss = loss + 0.5 * 0.005 * loss_proximal
            loss.backward()
            if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                                                  max_norm=self.clip_grad)
            optimizer.step()
