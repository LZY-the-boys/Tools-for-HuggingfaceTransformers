# Tools-for-Huggingface
|[English](https://github.com/LZY-the-boys/Tools-for-HuggingfaceTransformers/blob/main/README.md) | [中文](https://github.com/LZY-the-boys/Tools-for-HuggingfaceTransformers/blob/main/README-zh.md) |

关于 huggingface [transformers](https://github.com/huggingface/transformers) 库的一些魔改和笔记

## Codes

- 打字机模式 / 流式输出 (支持beam search, beam sample, sample, greedy) : `stream_generation.py` . 效果如下 (code in [Chinese-Vicuna](https://github.com/Facico/Chinese-Vicuna/)) :

https://user-images.githubusercontent.com/72137647/229296303-33fd239e-752a-4491-bfc1-555ee979eda6.mp4

- 关于 cdial-gpt (原始代码只支持 transformers==2.2.1) 在 transformers>=4 中的使用方法

## Notes

- huggingface transformers Trainer 的`report_to` 参数有一点要注意, `report_to=[]` 才能真正disable `wandb`, 而不是None; (this "feature" will be removed in transformers>=5.0)
- 关于bert的阅读笔记 [notes](https://github.com/LZY-the-boys/Tools-for-HuggingfaceTransformers/blob/main/Notes/bert.md)
- `peft-v0.3.0` 关于`lora`的源码阅读[笔记](https://github.com/LZY-the-boys/Tools-for-HuggingfaceTransformers/blob/main/Notes/peft-v0.3.0.md)