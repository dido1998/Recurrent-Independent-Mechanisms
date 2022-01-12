"""Logging functions to write to disk"""
import json
import logging
import time

from utils.util import flatten_dict


def _format_log(log):
    """format logs"""
    log = _add_time_to_log(log)
    return json.dumps(log)


def write_log(log):
    """This is the default method to write a log.
    It is assumed that the log has already been processed
     before feeding to this method"""
    get_logger().info(log)


def _add_time_to_log(log):
    log["timestamp"] = time.strftime('%I:%M%p %Z %b %d, %Y')
    return log


def read_log(log):
    """This is the single point to read any log message from the file.
     All the log messages are persisted as jsons"""
    try:
        data = json.loads(log)
    except json.JSONDecodeError as _:
        data = {
        }
    if data["type"] == "print":
        data = {

        }
    return data


def _format_custom_logs(keys, raw_log, _type="metric"):
    """Method to format the custom logs"""
    log = {}
    if keys:
        for key in keys:
            if key in raw_log:
                log[key] = raw_log[key]
    else:
        log = raw_log
    log["type"] = _type
    return _format_log(log), log


def write_message_logs(message, experiment_id=0):
    """"Write message logs"""
    kwargs = {"messgae": message, "experiment_id": experiment_id}
    log, _ = _format_custom_logs(keys=[], raw_log=kwargs, _type="print")
    write_log(log)


def write_trajectory_logs(trajectory, experiment_id=0):
    """"Write message logs"""
    kwargs = {"message": trajectory, "experiment_id": experiment_id}
    log, _ = _format_custom_logs(keys=[], raw_log=kwargs, _type="trajectory")
    write_log(log)


def write_config_log(config):
    """Write config logs"""
    config_to_write = json.loads(config.to_json())
    log, _ = _format_custom_logs(keys=[], raw_log=config_to_write, _type="config")
    write_log(log)


def write_metric_logs(metric):
    """Write metric logs"""
    keys = []
    log, _ = _format_custom_logs(keys=keys, raw_log=flatten_dict(metric), _type="metric")
    write_log(log)


def write_metadata_logs(**kwargs):
    """Write metadata logs"""
    log, _ = _format_custom_logs(keys=["best_epoch_index"], raw_log=kwargs, _type="metadata")
    write_log(log)


def pprint(config):
    """pretty print"""
    print(json.dumps(config, indent=4))


def set_logger(config):
    """Modified from
    https://docs.python.org/3/howto/logging-cookbook.html"""
    logger = logging.getLogger("default_logger")
    logger.setLevel(logging.INFO)
    # create file handler which logs all the messages
    logger_file_path = "{}/{}".format(config.folder_log, "log.txt")
    file_handler = logging.FileHandler(logger_file_path)
    file_handler.setLevel(logging.INFO)
    # create console handler with a higher log level
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def get_logger():
    """get logger"""
    return logging.getLogger("default_logger")