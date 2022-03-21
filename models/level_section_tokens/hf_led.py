from dataclasses import dataclass, field

from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)

from transformers import LEDForConditionalGeneration


@dataclass
class HuggingFaceLEDModelConfig(FairseqDataclass):
    hf_model_name: str = field(
        default='allenai/led-large-16384'
    )
    gradient_checkpointing: bool = field(
        default=False
    )


@register_model("hf_led", dataclass=HuggingFaceLEDModelConfig)
class HuggingFaceLEDModel(BaseFairseqModel):
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    @classmethod
    def build_model(cls, args, task):
        hf_model = LEDForConditionalGeneration.from_pretrained(args.hf_model_name, gradient_checkpointing=args.gradient_checkpointing, use_cache=False)
        return cls(hf_model)

    def forward(self, **kwargs):
        return self.hf_model(**kwargs)
