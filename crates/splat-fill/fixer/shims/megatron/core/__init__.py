"""Single-GPU stand-in for megatron.core: cosmos_predict2 only asks whether model parallelism is initialised."""
import dataclasses

from . import parallel_state  # noqa: F401


@dataclasses.dataclass
class ModelParallelConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = False
