# Tools-for-HuggingfaceTransformers
|[English](https://github.com/LZY-the-boys/Tools-for-HuggingfaceTransformers/blob/main/README.md) | [中文](https://github.com/LZY-the-boys/Tools-for-HuggingfaceTransformers/blob/main/README-zh.md) |

关于 huggingface [transformers](https://github.com/huggingface/transformers) 库的一些魔改

## Codes

- 打字机模式 / 流式输出 (支持beam search, beam sample, sample, greedy) : `stream_generation.py` . 效果如下 (code in [Chinese-Vicuna](https://github.com/Facico/Chinese-Vicuna/)) :

https://user-images.githubusercontent.com/72137647/229296303-33fd239e-752a-4491-bfc1-555ee979eda6.mp4

- 关于 cdial-gpt (原始代码只支持 transformers==2.2.1) 在 transformers>=4 中的使用方法

## Notes

- huggingface transformers Trainer `report_to` is badly designed, you need to set `report_to=[]` in trainer args to avoid the automatically `wandb`; this "feature" will be removed in transformers>=5.0

