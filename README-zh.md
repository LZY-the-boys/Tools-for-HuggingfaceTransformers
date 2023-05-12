# Tools-for-HuggingfaceTransformers
|[English](https://github.com/LZY-the-boys/Tools-for-HuggingfaceTransformers/blob/master/README.md) | [中文](https://github.com/LZY-the-boys/Tools-for-HuggingfaceTransformers/blob/master/docs/README-zh.md)

Custom tools for huggingface [transformers](https://github.com/huggingface/transformers)

## Codes

- streamly generation/ generation in typewriter mode: `stream_generation.py` . It supports the following effects (code in [Chinese-Vicuna](https://github.com/Facico/Chinese-Vicuna/)) :

https://user-images.githubusercontent.com/72137647/229296303-33fd239e-752a-4491-bfc1-555ee979eda6.mp4

- use cdial gpt (whose checkpoint is only trained in transformers==2.2.1) in transformers>=4 

## Notes

- huggingface transformers Trainer `report_to` is badly designed, you need to set `report_to=[]` in trainer args to avoid the automatically `wandb`; this "feature" will be removed in transformers>=5.0
