from torch.utils.tensorboard import SummaryWriter


def log_scalars(sum_writer: SummaryWriter, metric_dict, step):
    for k, v in metric_dict.items():
        sum_writer.add_scalar(k, v, step)
