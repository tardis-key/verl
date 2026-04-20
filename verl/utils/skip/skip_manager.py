from typing import Callable, Any, Optional, List
from verl.utils.skip.skip_rollout import SkipRollout
import functools

class BaseSkip:
    actions : List[str]
    action : str
    skip_functions : dict[str, Callable]
    preconditions : dict[str, Callable]

    @classmethod
    def meet_precondition(cls) -> bool:
        return cls.preconditions[cls.action]()
    
    @classmethod
    def warp_function(cls) -> Any:
        return cls.skip_functions[cls.action]()
    
    @classmethod
    def prepare_data(cls, result, *args, **kwargs) -> Any:
        pass

    @classmethod
    def always_true(self) -> bool:
        return True

class SkipManager:
    skip_classes : dict[str, BaseSkip] = {
        "generate": SkipRollout
    }
    config : Any
    step : int

    @classmethod
    def int(cls, config):
        cls.config = config
    
    @classmethod
    def annotate(cls, role: Optional[str] = None, **kwargs_outer) -> Callable:
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs_inner):
                if cls.config.role.enable and cls.step in cls.config.role.steps:
                    if cls.skip_classes[role].meet_precondition():
                        return cls.skip_classes[role].warp_function()
                    else:
                        result = func(*args, **kwargs_inner)
                        cls.skip_classes[role].prepare_data(result, *args, **kwargs_inner)
                        return result
                else:
                    return func(*args, **kwargs_inner)
            return wrapper
        return decorator