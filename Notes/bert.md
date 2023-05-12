# BERT

hugginface transformers BERT源码阅读

## reference

- https://zhuanlan.zhihu.com/p/360988428 
- https://zhuanlan.zhihu.com/p/363014957 
- http://fancyerii.github.io/2019/03/09/bert-codes/

## NOTEs

### sec1. tokenizer

和BERT有关的Tokenizer主要写在`tokenization_bert.py`和`tokenization_bert_fast.py` 中 这两份代码分别对应基本的`BertTokenizer`，以及不进行token到index映射的`BertTokenizerFast` 

- truncation截断逻辑 (`truncation=True`)： 将句子截断为`max_length-2` 因为要添加[cls]和[sep]
- pad填充逻辑：向右或者向左，向左一般是inference的时候避免输入部分和待生成部分被pad分开
- tokenize逻辑：
    - `BasicTokenizer`: 按空格分割句子，并处理是否统一小写，以及清理非法字符。
    对于中文字符，通过预处理（加空格）来按字分割
    - `WordPieceTokenizer` 在`BasicTokenizer`的基础上，进一步将词分解为子词（subword） 
    中文也有subword的，一般是比较生僻的词汇 
    但bert-base的tokenizer不会处理，因为在`basictokenizer`已经按字分开，wordpieces要么处理成[unk],要么不处理
    subword介于char和word之间，既在一定程度保留了词的含义，又能够照顾到英文中单复数、时态导致的词表爆炸和未登录词的OOV（Out-Of-Vocabulary）问题，将词根与时态词缀等分割出来，从而减小词表，也降低了训练难度, 有下面几种算法
        - BPE算法(GPT2 Roberta): 基于最高频字节对生成subword
        - WordPiece算法(Bert)：基于概率生成新的subword
        - ULM算法 和WordPiece一样是基于语言模型的，引入了一个假设：所有subword的出现都是独立的，并且subword序列由subword出现概率的乘积产生
    - `BertTokenizerFast` 更快，因为使用了 基于 RUST的[tokenizer](https://github.com/huggingface/tokenizers) 库，所以多线程更好

### sec2. model

- BERTbase（L=12，H = 768，A = 12，Total Parameters = 110M）
- BERTlarge（L = 24，H = 1024，A = 16，Total Parameters = 340M）
- BertTokenizer:
```
PreTrainedTokenizerFast(name_or_path='bert-base-chinese', vocab_size=21128, model_max_len=512, is_fast=True, padding_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})
=> nn.Embedding(21128, 768)
```


Model本体写在 `modeling_bert.py` 

- `BertLayer` 中的`apply_chunking_to_forward`是一个节约显存的技术——包装了一个切分小batch或者低维数操作的功能：这里参数`chunk_size`其实就是切分的batch大小，而`chunk_dim`就是一次计算维数的大小，最后拼接起来返回。

    - 参数`chunk_size`其实就是切分的batch大小，而`chunk_dim`就是一次计算维数的大小，最后拼接起来返回。不过，在默认操作中不会特意设置这两个值（在源代码中默认为0和1），所以会直接等效于正常的forward过程。

- `BertEncoder` 利用**gradient checkpointing**技术以降低训练时的显存占用。[torch实现](https://pytorch.org/docs/stable/checkpoint.html)

- `BertEmbedding`
    - **word_embeddings**，上文中subword对应的嵌入。
    - **token_type_embeddings**[段嵌入]，区分不同句子
    - **position_embeddings**，句子中每个词的位置嵌入，用于区别词的顺序。和transformer论文中的设计不同，这一块是训练出来的，而不是通过Sinusoidal函数计算得到的固定嵌入。一般认为这种实现不利于拓展性 https://github.com/google-research/bert/issues/58
    - 三个embedding不带权重相加，并通过一层LayerNorm+dropout后输出 [为什么用layerNorm](https://www.zhihu.com/question/395811291/answer/1260290120)

```python
>>> embedding = nn.Embedding(10, 3) # 输入是token对应词嵌入的idx
>>> input = torch.tensor([[1,1,1,1],[2,2,2,2]]) #torch.Size([2, 4])
>>> embedding(input) # torch.Size([2, 4, 3])
tensor([[[ 0.4306,  1.3666, -0.2756],
        [ 0.4306,  1.3666, -0.2756],
        [ 0.4306,  1.3666, -0.2756],
        [ 0.4306,  1.3666, -0.2756]],
        [[-0.1105, -0.4506, -1.2167],
        [-0.1105, -0.4506, -1.2167],
        [-0.1105, -0.4506, -1.2167],
        [-0.1105, -0.4506, -1.2167]]], grad_fn=<EmbeddingBackward>)
```

```python
>>> input= torch.arange(12, dtype=torch.long).view(-1, 1)-torch.arange(12, dtype=torch.long).view(1, -1)+12-1
tensor([[11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0],
        [12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1],
        [13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2],
        [14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3],
        [15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4],
        [16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5],
        [17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6],
        [18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7],
        [19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8],
        [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9],
        [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10],
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]])
>>> embedding(input).size() # 把每个数(idx)换成对应的emb[idx]=3d-vector
torch.Size([12, 12, 3])
```
  ​      
- `BertSelfAttention`

    - `prune_heads`: 剪枝操作

    - 这些注意力头，众所周知是并行计算的，所以上面的query、key、value三个权重是唯一的，所有heads是“拼接”起来。(12个头，768分为12个部分，每部分的矩阵768\*64，因为后面是拼接操作，所以可以直接用一个768\*768的矩阵实现 这样一个矩阵就实现了多头运算)


- FFN

    - 为什么要加ffn： https://arxiv.org/abs/2103.03404
    - gelu: 这里的激活函数默认实现为`gelu`（Gaussian Error Linerar Units(GELUS）： ![[公式]](https://www.zhihu.com/equation?tex=GELU%28x%29%3DxP%28X%3C%3Dx%29%3Dx%CE%A6%28x%29+) 

        ```python
        0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        ```

- `PreTrainedModel.tie_weights()`所有huggingface实现的PLM的word embedding和masked language model的预测权重在初始化过程中都是共享的


  - 指输入的word embedding的词表(nn.Embedding)和BertLMPredictionHead类中的self.decoder(nn.Linear)两个weight应该tie(tie的话应该是词表和这里的self.decoder同步梯度更新)或者clone(clone应该只是初始值一样，词表和self.decoder后面是分别更新)

- `BertForPreTraining` -> `BertPreTrainedModel` -> `PreTrainedModel`

  - `BertModel`

  - `BertPreTrainingHeads` -> `BertLMPredictionHead`

  - `BertForMaskedLM`：只进行MLM任务的预训练, mlm的逻辑是：

    ```
    Input:
    the man [MASK1] to [MASK2] store
    Label:
    [MASK1] = went; [MASK2] = store
    ```
    - 80%的概率，将token替换为特殊符号<mask>
    - 10%的概率，将token随机替换为vocabulary中的某个token (通过将token进行随机替换，给模型增加噪声，使得模型的泛化能力更强。)
    - 10%的概率，保持原token不变 (因为在之后的fine-tuning阶段，数据集中不会出现这些人造的<mask>标记，避免造成预训练的数据集和fine-tuning的数据集不匹配的情况)
    
    MLM（Masked Language Model）任务在预训练和微调时的不一致，也就是预训练出现了[MASK]而下游任务微调时没有[MASK]，是经常被吐槽的问题，很多工作都认为这是影响BERT微调性能的重要原因，并针对性地提出了很多改进，如[XL-NET](https://arxiv.org/abs/1906.08237)、[ELECTRA](https://arxiv.org/abs/2003.10555)、[MacBERT](https://arxiv.org/abs/2004.13922)等。

    - <mask>和直接attention mask -10000屏蔽的区别：<mask>会参与attention的计算 (占位符)
    - [Dropout视角下的MLM和MAE：一些新的启发](https://spaces.ac.cn/archives/8770)
    - bert-wwm和ernie1.0 span-bert就是对一个span进行mask，另外T5里面也参考了span-bert的做法，针对一个span进行mask，并预测被mask的span中的所有词

  - `BertLMHeadModel`：这个和上一个的区别在于，这一模型是**作为decoder运行**的版本；

    同样基于`BertOnlyMLMHead`；

  - `BertForNextSentencePrediction`：只进行NSP任务的预训练。

    基于`BertOnlyNSPHead`，内容就是一个线性层

- `BertForSequenceClassification` 分类任务，比如GLUE benchmark的各个任务 句子分类的输入为句子（对），输出为单个分类标签。
    ```
    BertForSequenceClassification(
        (bert): BertModel(
        (embeddings): BertEmbeddings(
            (word_embeddings): Embedding(21128, 768, padding_idx=0)
            (position_embeddings): Embedding(512, 768)
            (token_type_embeddings): Embedding(2, 768)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): BertEncoder_pagerank(
            (layer): ModuleList(
            (0-11): BertLayer(
                (attention): BertAttention(
                (self): BertSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                )
                (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
                )
                (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
                )
            )
            )
        )
        (pooler): BertPooler(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (activation): Tanh()
        )
        )
        (dropout): Dropout(p=0.1, inplace=False)
        (classifier): Linear(in_features=768, out_features=2, bias=True)
    )
    ```
- `BertForMultipleChoice` 用于多项选择，如RocStories/SWAG任务 多项选择任务的输入为一组分次输入的句子，输出为选择某一句子的单个标签
- `BertForTokenClassification` 序列标注（词分类），如NER任务。输入为单个句子文本，输出为每个token对应的类别标签。
- `BertForQuestionAnswering` 解决问答任务，例如SQuAD任务 

    ```
    BertForQuestionAnswering(
        (bert): BertModel(
        (embeddings): BertEmbeddings(
            (word_embeddings): nn.Embedding(vocab_size=21128, hidden_size=768, padding_idx=0)
            (position_embeddings): Embedding(max_position_embeddings=512, hidden_size=768)
            (token_type_embeddings): Embedding(type_vocab_size=2, hidden_size=768)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
            # 参数数：  (21128*768+512*768+2*768+2*768)
            # 维度： (batchsize,seq_length) -> (batchsize,seq_length,hidden_size)
        )

        (encoder): BertEncoder(
            (layer): ModuleList(
            (0-11): BertLayer( # 12层
                (attention): BertAttention(
                (self): BertSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                # [维度变化] b= batch_size, l= seq_length, s = hidden_size = num_head * head_size = h*d
                # qkv矩阵变换 (b,l,s) * (s,s) -> (b,l,s)
                # (多头理解: (b,l,s)*(s,h,d) -> (b,l,h,d) -> (b,l,s))
                # a = qk相乘 (b,l,s) * (b,s,l) -> (b,l,l)
                # av相乘 (b,l,l) * (b,l,s) -> (b,l,s)
                (output): BertSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                # 输出投影变换 (b,l,s) * (s,s) -> (b,l,s)
                # 计算量：b*(4 l s^2+2 l^2 s) (l<=512,s=768)
                # 参数数 4*768*768+2*768
                )
                
                (intermediate): BertIntermediate(
                    (dense): Linear(in_features=768, out_features=3072, bias=True)
                )
                (output): BertOutput(
                    (dense): Linear(in_features=3072, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                # 2次线性变换 (b,l,s) * (s,4s) -> (b,l,4s); (b,l,4s) * (4s,s) -> (b,l,s)
                # 计算量 b*(8 l s^2)
                # 参数量 8*768*768+2*768
            )
        )
        )
        # 每个token pos都会对应输出一个hidden_size的向量 (768 in BERT Base).对于分类任务，我们一般只用第一个cls输入
        (qa_outputs): Linear(in_features=768, out_features=2, bias=True)
    )
    )
    参数估算：(21128*768+512*768+2*768+2*768) + 12*(4*768*768+8*768*768+4*768) + 2*768 = 101595648
    ```


## 使用逻辑

- 在2023前，做NLU经常用BERT，但是做NLG以及做seq2seq的一般是GPT2和T5
- 训练的时候Decoder端的self-att和cross-att都是直接一遍过的 只有casual mask保证auto-regressive。
  - BART Attention事实上有4种情况： Encoder端 Decoder端(Train Generate)(masked self-att, cross att)
- 在pred的时候也就是`generate`的时候，会用到`past key values`
  - 自回归模型，我们输入一个开始符号<sos>，模型预测下一个词也就是第一个词w1，我们再把<sos>w1喂进去，模型预测w2，以此类推
  - 对于decoder，对于长度为n的预测，模型需要从下而上跑n遍，左端的K、V可以复用，用`past key values`  模型的`is_decoder=True`