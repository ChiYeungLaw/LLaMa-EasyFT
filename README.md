## LLaMa-EasyFT: A Toolkit for Fine-Tuning Large Language Models with LoRA and DeepSpeed

<h2 id="usage">Usage</h2>

- Setup environment:
```bash
git clone https://github.com/ChiYeungLaw/LLaMa-EasyFT.git
conda create -n llamaLora python=3.10
conda activate llamaLora
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
cd ..
pip install -r requirements.txt
```

- Alpaca Training data is avaliable:
```bash
data/alpaca_data.json
```

- Training LLaMA-7B:
```bash
bash finetune.sh
```


- Inference
```bash
bash generate.sh
```


## Thanks For

This toolkit are based on many great open source projects:

[Meta AI LLaMA](https://arxiv.org/abs/2302.13971v1), [Huggingface Transformers Llama](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama), [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html), [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)
, and [LLaMa-X](https://github.com/AetherCortex/Llama-X)




