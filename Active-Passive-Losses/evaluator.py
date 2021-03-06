import time
import torch
import os
from util import log_display, accuracy, AverageMeter

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

import json
import uuid
pred_list = []
identifier = str(uuid.uuid4())

class Evaluator():
    def __init__(self, data_loader, logger, config, name='Evaluator', metrics='classfication', summary_writer=None):
        self.data_loader = data_loader
        self.logger = logger
        self.name = name
        self.summary_writer = summary_writer
        self.step = 0
        self.config = config
        self.log_frequency = config.log_frequency
        self.loss_meters = AverageMeter()
        self.acc_meters = AverageMeter()
        self.acc5_meters = AverageMeter()
        self.report_metrics = self.classfication_metrics if metrics == 'classfication' else self.regression_metrics
        return

    def log(self, epoch, GLOBAL_STEP):
        display = log_display(epoch=epoch,
                              global_step=GLOBAL_STEP,
                              time_elapse=self.time_used,
                              **self.logger_payload)
        self.logger.info(display)

    def eval(self, epoch, GLOBAL_STEP, model, criterion, noise_rate, removal_rate, repeat_rate, model_name):
        for i, (images, labels) in enumerate(self.data_loader):
            self.eval_batch(x=images, y=labels, model=model, criterion=criterion)
        self.log(epoch, GLOBAL_STEP)

        fault_type = "mislabel"
        if (removal_rate > 0):
            noise_rate = removal_rate
            fault_type = "remove"
        elif (repeat_rate > 0):
            noise_rate = repeat_rate
            fault_type = "repeat"
        elif (noise_rate > 0):
            fault_type = "mislabel"
        else:
            noise_rate = 0
            fault_type = "golden"

        dataset_name = self.config.dataset.dataset_type.lower()

        with open("./injection/" + dataset_name + "/apl-" + model_name + "-" + fault_type + str(noise_rate) + "-" + str(identifier), "w") as w_file:
            json.dump(pred_list, w_file)

        pred_list.clear()

        return

    def eval_batch(self, x, y, model, criterion):
        model.eval()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        start = time.time()
        with torch.no_grad():
            pred = model(x)

            pred_results = torch.argmax(pred, dim=1)
            pred_temp = pred_results.cpu().tolist()
            pred_list.extend(pred_temp)

            loss = criterion(pred, y)
        end = time.time()
        self.time_used = end - start
        self.step += 1
        self.report_metrics(pred, y, loss)
        return

    def classfication_metrics(self, x, y, loss):
        dataset_name = self.config.dataset.dataset_type.lower()
        if dataset_name == "pneumonia":
            acc, acc5 = accuracy(x, y, topk=(1, 2))
        else:
            acc, acc5 = accuracy(x, y, topk=(1, 5))
        self.loss_meters.update(loss.item(), y.shape[0])
        self.acc_meters.update(acc.item(), y.shape[0])
        self.acc5_meters.update(acc5.item(), y.shape[0])
        self.logger_payload = {"acc": acc,
                               "acc_avg": self.acc_meters.avg,
                               "top5_acc": acc5,
                               "top5_acc_avg": self.acc5_meters.avg,
                               "loss": loss,
                               "loss_avg": self.loss_meters.avg}

        if self.summary_writer is not None:
            self.summary_writer.add_scalar(os.path.join(self.name, 'acc'), acc, self.step)
            self.summary_writer.add_scalar(os.path.join(self.name, 'loss'), loss, self.step)

    def regression_metrics(self, x, y, loss):
        diff = abs((x - y).mean().detach().item())
        self.loss_meters.update(loss.item(), y.shape[0])
        self.acc_meters.update(diff, y.shape[0])
        self.logger_payload = {"|diff|": diff,
                               "|diff_avg|": self.acc_meters.avg,
                               "loss": loss,
                               "loss_avg": self.loss_meters.avg}

        if self.summary_writer is not None:
            self.summary_writer.add_scalar(os.path.join(self.name, 'diff'), diff, self.step)
            self.summary_writer.add_scalar(os.path.join(self.name, 'loss'), loss, self.step)

    def _reset_stats(self):
        self.loss_meters.reset()
        self.acc_meters.reset()
        self.acc5_meters.reset()
