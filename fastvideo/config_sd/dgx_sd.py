import ml_collections
from ml_collections import config_dict
import importlib.util
import pathlib
import sys
import os

base_path = pathlib.Path(__file__).with_name("base.py")

# 2️⃣  为该文件创建一个模块“规格”（spec）
spec = importlib.util.spec_from_file_location("base", base_path)

# 3️⃣  根据 spec 生成空模块对象
base = importlib.util.module_from_spec(spec)

# 4️⃣  把模块登记到 sys.modules，使得其它地方 `import base` 也能找到
sys.modules["base"] = base

# 5️⃣  执行 base.py，把其中的代码填充到模块命名空间
spec.loader.exec_module(base)

config = ml_collections.ConfigDict()

def compressibility():
    config = base.get_config()

    config.pretrained.model = "CompVis/stable-diffusion-v1-4"

    config.num_epochs = 300
    config.save_freq = 50
    config.num_checkpoint_limit = 100000000

    # the DGX machine I used had 8 GPUs, so this corresponds to 8 * 8 * 4 = 256 samples per epoch.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    # this corresponds to (8 * 4) / (4 * 2) = 4 gradient updates per epoch.
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 4

    # prompting
    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    # rewards
    config.reward_fn = "jpeg_compressibility"

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }

    return config

def hps():
    config = compressibility()
    config.num_epochs = 300
    config.reward_fn = "aesthetic_score"

    # this reward is a bit harder to optimize, so I used 2 gradient updates per epoch.
    config.train.gradient_accumulation_steps = 8

    # the DGX machine I used had 8 GPUs, so this corresponds to 8 * 8 * 4 = 256 samples per epoch.
    config.sample.batch_size = 4

    # this corresponds to (8 * 4) / (4 * 2) = 4 gradient updates per epoch.
    config.train.batch_size = 4

    config.prompt_fn = "aes"
    config.chosen_number = 16
    config.num_generations = 16
    return config


def clip_score():
    """纯CLIP Score奖励配置"""
    config = base.get_config()
    
    config.pretrained.model = "CompVis/stable-diffusion-v1-4"
    config.num_epochs = 300
    config.save_freq = 50
    config.num_checkpoint_limit = 100000000
    
    # 奖励配置
    # config.reward_fn = config_dict.ConfigDict({"clip_score": 1.0})
    config.train.gradient_accumulation_steps = 8
    config.reward_fn = {"clip_score": 1.0}
    config.reward_types = ["clip_score"]
    
    # 训练参数调整（CLIP Score可能需要不同的参数）
    config.train.learning_rate = 1e-6  # 稍微保守的学习率
    config.train.gradient_accumulation_steps = 8
    
    # DGX 8 GPU配置
    config.sample.batch_size = 4
    config.sample.num_batches_per_epoch = 4
    
    # 训练batch配置
    config.train.batch_size = 2
    
    config.prompt_fn = "aes"
    config.prompt_fn_kwargs = {}
    config.num_generations = 16
    
    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }
    
    return config

def mixed_rewards():
    """混合奖励配置 - HPS + CLIP Score"""
    config = base.get_config()
    
    config.pretrained.model = "CompVis/stable-diffusion-v1-4"
    config.num_epochs = 400  # 混合奖励可能需要更多epoch
    config.save_freq = 50
    config.num_checkpoint_limit = 100000000
    
    # 混合奖励配置
    config.reward_fn = {
        "hps": 0.7,           # HPS权重
        "clip_score": 1.4,    # CLIP Score权重（更高）
    }
    
    # 训练参数调整
    config.train.learning_rate = 8e-6
    config.train.gradient_accumulation_steps = 8
    
    # DGX 8 GPU配置
    config.sample.batch_size = 4
    config.sample.num_batches_per_epoch = 4
    
    # 训练batch配置  
    config.train.batch_size = 2
    
    config.prompt_fn = "aes"
    config.prompt_fn_kwargs = {}
    config.num_generations = 16
    
    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }
    
    return config

def high_quality_mixed():
    """高质量混合配置 - HPS + CLIP + Aesthetic"""
    config = base.get_config()
    
    config.pretrained.model = "CompVis/stable-diffusion-v1-4"
    config.num_epochs = 500
    config.save_freq = 50
    config.num_checkpoint_limit = 100000000
    
    # 三重奖励组合
    config.reward_fn = {
        "hps": 1.0,
        "clip_score": 1.2,
        "aesthetic": 0.8,  # 添加美学评分
    }
    
    # 更保守的训练设置
    config.train.learning_rate = 3e-6
    config.train.gradient_accumulation_steps = 12
    config.train.clip_range = 5e-5  # 更小的clip range
    
    # DGX 8 GPU配置 - 减小batch size确保质量
    config.sample.batch_size = 2
    config.sample.num_batches_per_epoch = 6
    
    # 训练batch配置
    config.train.batch_size = 1
    
    config.prompt_fn = "aes"
    config.prompt_fn_kwargs = {}
    config.num_generations = 8
    
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
    
    return config

def aesthetic_only():
    """纯美学评分配置"""
    config = base.get_config()
    
    config.pretrained.model = "CompVis/stable-diffusion-v1-4"
    config.num_epochs = 300
    config.save_freq = 50
    config.num_checkpoint_limit = 100000000
    
    # 奖励配置
    config.reward_fn = {"aesthetic": 1.0}
    
    # 美学评分通常需要更小的学习率
    config.train.learning_rate = 2e-6
    config.train.gradient_accumulation_steps = 10
    
    # DGX 8 GPU配置（基于hps配置）
    config.sample.batch_size = 4
    config.sample.num_batches_per_epoch = 4
    
    # 训练batch配置
    config.train.batch_size = 4
    
    config.prompt_fn = "aes"
    config.prompt_fn_kwargs = {}
    config.chosen_number = 16
    config.num_generations = 12
    
    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }
    
    return config

def balanced_mixed():
    """均衡混合配置 - 所有奖励权重相等"""
    config = base.get_config()
    
    config.pretrained.model = "CompVis/stable-diffusion-v1-4"
    config.num_epochs = 350
    config.save_freq = 50
    config.num_checkpoint_limit = 100000000
    
    # 四种奖励均衡组合
    config.reward_fn = {
        "hps": 0.8,
        "clip_score": 0.8,
        "aesthetic": 0.8,
        "jpeg_compressibility": 0.6,  # 稍低权重
    }
    
    # 平衡的训练设置
    config.train.learning_rate = 6e-6
    config.train.gradient_accumulation_steps = 6
    
    # DGX 8 GPU配置
    config.sample.batch_size = 3
    config.sample.num_batches_per_epoch = 5
    
    config.train.batch_size = 2
    
    config.prompt_fn = "aes"
    config.prompt_fn_kwargs = {}
    config.num_generations = 15
    
    config.per_prompt_stat_tracking = {
        "buffer_size": 24,
        "min_count": 12,
    }
    
    return config

def get_config(name):
    return globals()[name]()
