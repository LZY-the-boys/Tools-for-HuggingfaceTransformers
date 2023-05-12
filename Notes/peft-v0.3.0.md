#### `Multi-LoRA`实现

效果

```
transformers.models.llama.modeling_llama.LlamaAttention
  (q_proj): Linear8bitLt(
    in_features=4096, out_features=4096, bias=False
    (lora_dropout): ModuleDict(
      (Medical_Alpaca): Dropout(p=0.05, inplace=False)
      (Chinese_Alpaca): Dropout(p=0.05, inplace=False)
    )
    (lora_A): ModuleDict(
      (Medical_Alpaca): Linear(in_features=4096, out_features=8, bias=False)
      (Chinese_Alpaca): Linear(in_features=4096, out_features=8, bias=False)
    )
    (lora_B): ModuleDict(
      (Medical_Alpaca): Linear(in_features=8, out_features=4096, bias=False)
      (Chinese_Alpaca): Linear(in_features=8, out_features=4096, bias=False)
    )
    (lora_embedding_A): ParameterDict()
    (lora_embedding_B): ParameterDict()
  )

transformers.models.llama.modeling_llama.LlamaMLP.gate_proj / down_proj / up_proj : 

Linear8bitLt(
    in_features=4096, out_features=11008, bias=False
    (lora_dropout): ModuleDict(
      (Chinese_Alpaca): Dropout(p=0.05, inplace=False)
    )
    (lora_A): ModuleDict(
      (Chinese_Alpaca): Linear(in_features=4096, out_features=8, bias=False)
    )
    (lora_B): ModuleDict(
      (Chinese_Alpaca): Linear(in_features=8, out_features=11008, bias=False)
    )
    (lora_embedding_A): ParameterDict()
    (lora_embedding_B): ParameterDict()
)
```

##### `forward`

就是拿MOE思想设计的，同时保存多个参数，用`adapter_name` 决定`forward`调用哪一个`adapter` , 用`set_adapter` 更改

```
output = (
self.lora_B[adapter_name](
self.lora_A[adapter_name](
self.lora_dropout[adapter_name](x))).to(expected_dtype)
* self.scaling[adapter_name])
```

##### `load_adapter` 和 `set_adapter`

```
-> peft.peft_model.PeftModel.load_adapter(path, adapter_name)
-> peft.tuners.lora.LoraModel.add_adapter(adapter_name, peft_config)
-> peft.tuners.lora.LoraModel._find_and_replace(adapter_name)
-> peft.tuners.lora.LoraModel._replace_module
    -> new_module = replace(adapter_name)
    -> old_module = new_module
```

##### `replace` 

逻辑是做如下的替换：

- `bnb.nn.Linear8bitLt` -> `Linear8bitLt` 这是8bit版本的带lora的线性层
- `torch.nn.Linear` -> `Linear`  forward调`F.linear`  fp16和fp32版本的带lora的线性层 
- `torch.nn.Embedding` -> `Embedding`  forward调`F.embedding`  带lora的embedding层
- `LoraLayer` (`Linear8bitLt` `Linear` `Embedding` 的父类)-> `update_layer`   不会修改`adapter_name`  上面三个则会修改，所以只有第一次初始化的lora会被自动激活 ==(注意以module为单位，而不是以模型为单位)==

##### `update_layer`

```
    Linear8bitLt.r: {'Medical_Alpaca': 8, 'Chinese_Alpaca': 8}
    Linear8bitLt.lora_alpha: {'Medical_Alpaca': 16, 'Chinese_Alpaca': 16}
    Linear8bitLt.lora_A: ModuleDict(
      (Medical_Alpaca): Linear(in_features=4096, out_features=r, bias=False)
      (Chinese_Alpaca): Linear(in_features=4096, out_features=r, bias=False)
    )
    scaling[name] = lora_alpha / r
-> reset_lora_parameters
	A->kaiming_uniform B->zeros
```

##### `merge`

目前只有`Linear`实现了`merge`， 因此只能在fp16下合并参数
$$
Linear(x) = F.linear(x,w,b)=xw^T+b\\
Lora(x) = Linear(x)+scale*xA^TB^T\\=x(w^T+scale*A^TB^T)+b \\= F.linear(x,w+scale*BA,b)\\=Linear'(x)
$$
int8下由于bitsandbytes重载了矩阵计算，因此不一定有上式的等价关系，所以peft现在还没有实现int8下的merge

```
self.weight.data += (transpose(
self.lora_B[name].weight @ self.lora_A[name].weight,
self.fan_in_fan_out,
)* self.scaling[self.active_adapter])
self.merged = True
```

