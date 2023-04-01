# Tools-for-HuggingfaceTransformers

custom tools for huggingface [transformers](https://github.com/huggingface/transformers)

## Codes

- streamly generation/ generation in typewriter mode: `stream_generation.py` ; `stream_generation_peft.py`
- use cdial gpt (whose checkpoint is only trained in transformers==2.2.1) in transformers>=4 

## Notes

- huggingface transformers Trainer `report_to` is badly designed, you need to set `report_to=[]` in trainer args to avoid the automatically `wandb`; this "feature" will be removed in transformers>=5.0

