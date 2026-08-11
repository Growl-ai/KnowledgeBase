import colorlog
import logging
from abc import ABC, abstractmethod
from typing import TypeVar, Optional
from processor.import_processor.config import ImportConfig, get_config
from processor.import_processor.exceptions import ImportProcessError
from utils.task_trace import add_running_task, add_done_task

T = TypeVar('T')

class BaseNode(ABC):
    name: str="base_node"
    def __init__(self, config: Optional[ImportConfig]=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) ->T:
        try:
            self.logger.info(f"----------{self.name} 开始----------")
            # 开始：记录节点运行状态
            add_running_task(state["task_id"], self.name)
            result = self.process(state)
            # 结束：记录节点完成状态
            add_done_task(state["task_id"], self.name)
            self.logger.info(f"----------{self.name} 结束----------")
            return result
        except Exception as e:
            self.logger.error(f"Error in {self.name}: {e}")
            raise ImportProcessError(message=str(e), node_name=self.name, cause=e)

    @abstractmethod
    def process(self, state:T) -> T:
        pass


    def log_step(self, step_name: str, message: str=""):
        log_msg = f"[{step_name}]: {message}"
        self.logger.info(log_msg)




def setup_logging(level: int=logging.INFO):
    """设置日志格式"""
    logger = logging.getLogger()
    logger.setLevel(level)
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    logger.handlers.clear()
    logger.addHandler(handler)

