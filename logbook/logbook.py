"""Wrapper over wandb api"""

#import wandb
from box import Box

from logbook import util as log_func
from utils.util import flatten_dict, make_dir


class LogBook():
    """Wrapper over comet_ml api"""

    def __init__(self, config):
        self.metrics_to_record = [
            "mode",
            "batch_idx",
            "epoch",
            "time",
            "correct",
            "loss",
            "kl_loss",
            "time_taken",
            "stove_pixel_loss_10"
        ] #+ [f"activation_frequency_{idx}" for idx in range(config.num_blocks)]
        self._experiment_id = 1
        config_dict = vars(config)

        flattened_config = flatten_dict(config_dict, sep="_")

        self.config = Box(config_dict)
        # logger_file_dir = "{}/{}".format(self.config.log_folder, self.config.id)
        if not config.should_resume:
            make_dir(self.config.folder_log)
        #make_dir(self.config.folder_log)
        log_func.set_logger(self.config)

        self.should_use_remote_logger = False

        if self.should_use_remote_logger:
            wandb.init(config=flattened_config,
                       project="ool",
                       name=self.config.id,
                       dir=self.config.folder_log,
                       entity="kappa")

        self.tensorboard_writer = None
        self.should_use_tb = False

        log_func.write_config_log(self.config)

    def log_metrics(self, dic, prefix, step):
        """Method to log metric"""
        formatted_dict = {}
        for key, val in dic.items():
            formatted_dict[prefix + "_" + key] = val
        if self.should_use_remote_logger:
            wandb.log(formatted_dict, step=step)

    def write_config_log(self, config):
        """Write config"""
        log_func.write_config_log(config)
        flatten_config = flatten_dict(config, sep="_")
        flatten_config['experiment_id'] = self._experiment_id

    def write_metric_logs(self, metrics):
        """Write Metric"""
        metrics['experiment_id'] = self._experiment_id
        log_func.write_metric_logs(metrics)
        flattened_metrics = flatten_dict(metrics, sep="_")
        #key = "activation_frequency"
        #activation_frequency = flattened_metrics.pop(key)
        #for idx, freq in enumerate(activation_frequency):
        #    flattened_metrics[f"{key}_{idx}"] = freq
        metric_dict = {
            key: flattened_metrics[key]
            for key in self.metrics_to_record if key in flattened_metrics
        }
        prefix = metric_dict.pop("mode")
        batch_idx = metric_dict["batch_idx"]
        self.log_metrics(dic=metric_dict,
                         prefix=prefix,
                         step=batch_idx)

        # if self.should_use_tb:
        #
        #     timestep_key = "num_timesteps"
        #     for key in set(list(metrics.keys())) - set([timestep_key]):
        #         self.tensorboard_writer.add_scalar(tag=key,
        #                                            scalar_value=metrics[key],
        #                                            global_step=metrics[timestep_key])

    def write_image(self, img, mode, step, caption):
        if self.should_use_remote_logger:
            return wandb.log({f"{mode}_{caption}":
                              [wandb.Image(img, caption = str(step))]}, step=step)

    def write_compute_logs(self, **kwargs):
        """Write Compute Logs"""
        kwargs['experiment_id'] = self._experiment_id
        log_func.write_metric_logs(**kwargs)
        metric_dict = flatten_dict(kwargs, sep="_")

        num_timesteps = metric_dict.pop("num_timesteps")
        self.log_metrics(dic=metric_dict,
                         step=num_timesteps,
                         prefix="compute")

    def write_message_logs(self, message):
        """Write message"""
        log_func.write_message_logs(message, experiment_id=self._experiment_id)

    def write_metadata_logs(self, **kwargs):
        """Write metadata"""
        log_func.write_metadata_logs(**kwargs)
        # self.log_other(key="best_epoch_index", value=kwargs["best_epoch_index"])

    def watch_model(self, model):
        """Method to track the gradients of the model"""
        if self.should_use_remote_logger:
            wandb.watch(models=model, log="all")
