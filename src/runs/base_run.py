# Use 'Chain of Responsibility pattern' to define the run method for each run.

from abc import ABC, abstractmethod
from typing import Optional, Any


class BaseRunStep(ABC):
    @abstractmethod
    def set_step(self, base_run: 'BaseRunStep') -> 'BaseRunStep':
        pass

    @abstractmethod
    def run(self, request_config) -> Optional[None]:
        pass

class AbstractRunStep(BaseRunStep):

    _next_run_step: BaseRunStep = None

    def set_step(self, next_run_step: BaseRunStep) -> BaseRunStep:
        self._next_run_step = next_run_step
        return next_run_step

    @abstractmethod
    def run(self, **kwargs: Any) -> Optional[None]:
        if self._next_run_step:
            return self._next_run_step.run(kwargs)

        return None