# Introduction
************************************************************
This is a classification project based on pytorch,provides multi-gpu training, distillation training, QAT and PTQ.
All functions can be executed by modifying the yml file, and a rich customization interface is provided.

### Supported Projects
* https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
* https://arxiv.org/pdf/1503.02531.pdf

# Get Started
************************************************************
### Train a model
In this section, we will introduce how to train a model and where to view your training logs and models.

1.We train models on different datasets and backbone architectures through different yml files. Please refer to the comments in **_config/norm/example.yml_** for the specific parameter annotations.
You can query all currently supported network structures through the **_networks/__init__.py_** file and customize new networks
* To train MiniFASNetV1SE,run the following commend(with 4 GPUs)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=12345 train.py --config=config/norm/Head.yml
```
2.We provide the MiniFASNet series of QAT models. If you need QAT training for other models, you need to provide the corresponding quantization model instance yourself. Currently, QAT training only supports single-gpu
* To train QATMiniFASNetV1SE,run the following commend(with Single-GPU)
```
CUDA_VISIBLE_DEVICES=0 python train.py --config=config/qat/Head.yml
```

3.After finishing training a model, you will see a project folder under root. The trained model and log is saved in the folder named by the job starting time,eg,20240624_111913 for 11:19:13 on 2024-06-24.

### Test a model
During the testing phase,you can test it very easily using yml files.Both normal model and torch.jit.save quantized model are OK
```
python test.py --config=config/val/Head.yml
```
### Do PTQ
We also support post-quantization of the model, only the MiniFASNet series now
```
python do_ptq.py --config=config/ptq/Head.yml
```