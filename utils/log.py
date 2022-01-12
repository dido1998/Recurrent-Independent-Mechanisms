from collections import defaultdict
from datetime import datetime
import json
import logging
import numpy as np
#import git
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set

NORMAL_FORMATTER = logging.Formatter('%(levelname)s %(asctime)s: %(name)s: %(message)s')
JSON_FORMATTER = logging.Formatter('%(levelname)s::%(message)s')
FINGERPRINT = 'fingerprint.txt'
LOGFILE = 'log.txt'
EXP = 5
logging.addLevelName(EXP, 'EXP')


class Logger(logging.Logger):
    def __init__(self) -> None:
        # set log level to debug
        super().__init__('rainy', EXP)
        self._log_dir: Optional[Path] = None
        self.exp_start = datetime.now()

    def set_dir_from_script_path(
            self,
            script_path_: str,
            comment: Optional[str] = None,
            prefix: str = '',
    ) -> None:
        script_path = Path(script_path_)
        log_dir = script_path.stem + '-' + self.exp_start.strftime("%y%m%d-%H%M%S")
        if prefix:
            log_dir = prefix + '/' + log_dir
        try:
            repo = git.Repo(script_path, search_parent_directories=True)
            head = repo.head.commit
            log_dir += '-' + head.hexsha[:8]
        finally:
            pass
        log_dir_path = Path(log_dir)
        if not log_dir_path.exists():
            log_dir_path.mkdir()
        self.set_dir(log_dir_path, comment=comment)

    def set_dir(self, log_dir: Path, comment: Optional[str] = None) -> None:
        self._log_dir = log_dir

        def make_handler(log_path: Path, level: int) -> logging.Handler:
            if not log_path.exists():
                log_path.touch()
            handler = logging.FileHandler(log_path.as_posix())
            handler.setFormatter(JSON_FORMATTER)
            handler.setLevel(level)
            return handler
        finger = log_dir.joinpath(FINGERPRINT)
        with open(finger.as_posix(), 'w') as f:
            f.write('{}\n'.format(self.exp_start))
            if comment:
                f.write(comment)
        handler = make_handler(Path(log_dir).joinpath(LOGFILE), EXP)
        self.addHandler(handler)

    def set_stderr(self, level: int = EXP) -> None:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(NORMAL_FORMATTER)
        handler.setLevel(level)
        self.addHandler(handler)

    @property
    def log_dir(self) -> Optional[Path]:
        return self._log_dir

    def exp(self, name: str, msg: dict, *args, **kwargs) -> None:
        """
        For experiment logging. Only dict is enabled as argument
        """
        if not self.isEnabledFor(EXP):
            return
        delta = datetime.now() - self.exp_start
        msg['elapsed-time'] = delta.total_seconds()
        msg['name'] = name
        self._log(EXP, json.dumps(msg, sort_keys=True), args, **kwargs)  # type: ignore


def _load_log_file(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path.as_posix()) as f:
        lines = f.readlines()
    log = []
    for line in filter(lambda s: s.startswith('EXP::'), lines):
        log.append(json.loads(line[5:]))
    return log


class LogWrapper:
    """Wrapper of filterd log.
    """
    def __init__(
            self,
            name: str,
            inner: List[Dict[str, Any]],
            path: Optional[Path] = None,
    ) -> None:
        self.name = name
        self.inner = inner
        self._available_keys: Set[str] = set()
        self._path = path

    @property
    def unwrapped(self) -> List[Dict[str, Any]]:
        return self.inner

    def keys(self) -> Set[str]:
        if not self._available_keys:
            for log in self.inner:
                for key in log:
                    self._available_keys.add(key)
        return self._available_keys

    def get(self, key: str) -> List[Any]:
        if key not in self.inner[0]:
            raise KeyError(
                'LogWrapper({}) doesn\'t have the logging key {}. Available keys: {}'
                .format(self.name, key, self.keys())
            )
        return list(map(lambda d: d[key], self.inner))

    def is_empty(self) -> bool:
        return len(self.inner) == 0

    def __repr__(self) -> str:
        return 'LogWrapper({}, {})'.format(self._path, self.name)

    def __getitem__(self, key: str) -> List[Any]:
        return self.get(key)


class ExperimentLog:
    """Structured log file.
       Used to get graphs or else from rainy log files.
    """
    def __init__(self, file_or_dir_name: str) -> None:
        path = Path(file_or_dir_name)
        if path.is_dir():
            log_path = path.joinpath(LOGFILE)
            self.fingerprint = path.joinpath(FINGERPRINT).read_text()
        else:
            log_path = path
            self.fingerprint = ''
        self.log = _load_log_file(log_path)
        self._available_keys: Set[str] = set()
        self.log_path = log_path

    def keys(self) -> Set[str]:
        if not self._available_keys:
            for log in self.log:
                self._available_keys.add(log['name'])
        return self._available_keys

    def get(self, key: str) -> LogWrapper:
        log = LogWrapper(
            key,
            list(filter(lambda log: log['name'] == key, self.log)),
            self.log_path
        )
        if log.is_empty():
            raise KeyError(
                '{} doesn\'t have the key {}. Available keys: {}'
                .format(self, key, self.keys())
            )
        return log

    def plot_reward(self, batch_size: int, max_steps: int = int(2e7), title: str = '') -> None:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as e:
            print('plot_reward need matplotlib installed')
            raise e
        tlog = self.get('train')
        x, y = 'update-steps', 'reward-mean'
        plt.plot(np.array(tlog[x]) * batch_size, tlog[y])
        tick_fractions = np.array([0.1, 0.2, 0.5, 1.0])
        ticks = tick_fractions * max_steps
        MILLION = int(1e6)
        if max_steps >= MILLION:
            tick_names = ["{}M".format(int(tick / 1e6)) for tick in ticks]
        else:
            tick_names = ["{}".format(int(tick)) for tick in ticks]
        plt.xticks(ticks, tick_names)
        plt.title(title)
        plt.xlabel('Frames used for training')
        plt.ylabel(y)
        plt.show()

    def __getitem__(self, key: str) -> LogWrapper:
        return self.get(key)

    def __repr__(self) -> str:
        return 'ExperimentLog({})'.format(self.log_path.as_posix())


class ExpStats:
    """Statictics of loss
    """
    def __init__(self) -> None:
        self.inner: Dict[str, List[float]] = defaultdict(list)

    def update(self, d: Dict[str, float]) -> None:
        for key in d.keys():
            self.inner[key].append(d[key])

    def report_and_reset(self) -> Dict[str, float]:
        res = {}
        for k, v in self.inner.items():
            res[k] = np.array(v).mean()
            v.clear()
        return res
