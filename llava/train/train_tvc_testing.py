from llava.model import *
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import transformers
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from peft import LoraConfig, get_peft_model
import torch

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.model.tcv.modelling_tcv import TCVLlamaForCausalLM
from llava.model.tcv.tcv_configs import TCVLlamaConfig
from PIL import Image


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    version: Optional[str] = field(default="plain")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=True)
    vision_tower: Optional[str] = field(default="openai/clip-vit-large-patch14-336")

    vision_tower_text_encoder: Optional[str] = field(default="google-bert/bert-base-uncased") # Shapiro
    vision_tower_text_projector_type: Optional[str] = field(default='mlp2x_gelu') # Shapiro
    text_conditioned_vision_tower: bool = field(default=True)
    
    mm_vision_select_layer: Optional[int] = field(default=-2)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    text_select_feature: Optional[str] = field(default="all") #pool
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    
    llm_dora_enable: bool = False # Shapiro
    llm_lora_enable: bool = False
    llm_lora_r: int = 64
    llm_lora_alpha: int = 16
    llm_lora_dropout: float = 0.05
    llm_lora_weight_path: str = ""
    llm_lora_bias: str = "none"
    
    vit_dora_enable: bool = False # Shapiro
    vit_lora_enable: bool = False
    vit_lora_r: int = 64
    vit_lora_alpha: int = 16
    vit_lora_dropout: float = 0.05
    vit_lora_weight_path: str = ""
    vit_lora_bias: str = "none"
    
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    
@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 vit_text_tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.vit_text_tokenizer = vit_text_tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_llama_2(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        #TODO
        data_dict['input_prompts'] = [self.vit_text_tokenizer.encode(i['value'].replace("<image>", "").strip(), return_tensors='pt') for i in self.list_data_dict[i]['conversations'] if i['from'] == 'human' ][0].squeeze(0)
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    vit_text_tokenizer: transformers.PreTrainedTokenizer
    device: torch.device

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, vit_text_input_ids = tuple([instance[key] for instance in instances]
                                                        for key in ("input_ids", "labels", "input_prompts"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        try : 
            vit_text_input_ids = torch.nn.utils.rnn.pad_sequence(   vit_text_input_ids,
                                                                    batch_first=True,
                                                                    padding_value=self.vit_text_tokenizer.pad_token_id)
            vit_text_input_ids = vit_text_input_ids[:, :self.vit_text_tokenizer.model_max_length]
        except:
            raise NotImplementedError
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            vit_text_input_ids = vit_text_input_ids,
            vit_text_attention_mask = vit_text_input_ids.ne(self.vit_text_tokenizer.pad_token_id)
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch




model_args = ModelArguments()
data_args = DataArguments()
training_args = TrainingArguments(output_dir = "./checkpoints/llava-v1.5-13b-lora" )

# model_args 
model_args.model_name_or_path = 'meta-llama/Llama-2-7b-chat-hf'
model_args.vision_tower = "openai/clip-vit-large-patch14-336"
model_args.vision_tower_text_encoder = "google-bert/bert-base-uncased"
model_args.vision_tower_text_projector_type = 'mlp2x_gelu' # Shapiro
model_args.text_conditioned_vision_tower = True
model_args.mm_vision_select_layer = -2
model_args.pretrain_mm_mlp_adapter = None
model_args.tune_mm_mlp_adapter = True
model_args.mm_projector_type = 'mlp2x_gelu'
model_args.mm_use_im_start_end = False
model_args.mm_use_im_patch_token = False
model_args.mm_patch_merge_type = 'flat'
model_args.mm_vision_select_feature = "patch"
model_args.text_select_feature = "all"



# data_args
data_args.data_path = "/scratch/mt/new-structure/experiments/ashapiro/LLaVA/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
data_args.image_aspect_ratio = "pad"
data_args.lazy_preprocess = True
data_args.image_folder = "/scratch/mt/new-structure/experiments/ashapiro/LLaVA/playground/data/LLaVA-Pretrain/images"



# training_args
training_args.llm_dora_enable = True
training_args.llm_lora_r = 128
training_args.llm_lora_alpha = 256 

training_args.vit_dora_enable = True
training_args.vit_lora_r = 128
training_args.vit_lora_alpha = 128 

    
training_args.mm_projector_lr = 2e-5 
training_args.group_by_modality_length = True 
training_args.model_max_length = 2048 
training_args.bf16 = True 
training_args.output_dir = "./checkpoints/llava-v1.5-13b-lora" 
training_args.num_train_epochs = 1 
training_args.per_device_train_batch_size = 3 
training_args.per_device_eval_batch_size = 4 
training_args.gradient_accumulation_steps = 1 
training_args.evaluation_strategy = "no" 
training_args.save_strategy = "steps" 
training_args.save_steps = 50000 
training_args.save_total_limit = 1 
training_args.learning_rate = 2e-4 
training_args.weight_decay = 0. 
training_args.warmup_ratio = 0.03 
training_args.lr_scheduler_type = "cosine" 
training_args.logging_steps = 1 
training_args.tf32 = True 
# training_args.gradient_checkpointing =  True
training_args.gradient_checkpointing =  False 
training_args.dataloader_num_workers = 4 
training_args.report_to = []

training_args.dataloader_num_workers = 1
training_args.dataloader_pin_memory  = True
training_args.dataloader_persistent_workers = True


config = TCVLlamaConfig()
model = TCVLlamaForCausalLM(config)
model.to(device = torch.device("cuda:0"), dtype=torch.bfloat16)



llm_lora_config = LoraConfig(
    use_dora = True,
    r=training_args.llm_lora_r,
    lora_alpha=training_args.llm_lora_alpha,
    target_modules=find_all_linear_names(model.llm),
    lora_dropout=training_args.llm_lora_dropout,
    bias=training_args.llm_lora_bias,
    task_type="CAUSAL_LM",
)

vit_lora_config = LoraConfig(
    use_dora = True,
    r=16,
    lora_alpha=16,
    target_modules=find_all_linear_names(model.tcv.vision_model),
    lora_dropout=0.1,
    bias="none"
)

model.wrap_peft(
    
    llm_lora_config= llm_lora_config,
    vit_lora_config= vit_lora_config
)


llm_tokenizer = transformers.AutoTokenizer.from_pretrained(
    model.config.llm_config._name_or_path,
    cache_dir=training_args.cache_dir,
    model_max_length=training_args.model_max_length,
    padding_side="right",
    use_fast=False,
)
if not llm_tokenizer.pad_token:
    llm_tokenizer.pad_token = llm_tokenizer.unk_token


vit_tokenizer = transformers.AutoTokenizer.from_pretrained(
    model.config.tcv_config.text_config._name_or_path,
    cache_dir=training_args.cache_dir,
    model_max_length=training_args.model_max_length,
    padding_side="right",
    use_fast=False,
)

if not vit_tokenizer.pad_token:
    vit_tokenizer.pad_token = vit_tokenizer.unk_token



conversation_lib.default_conversation = conversation_lib.conv_templates["llama_2"]

data_args.image_processor = model.tcv.image_processor
data_args.is_multimodal = True

model.config.image_aspect_ratio = data_args.image_aspect_ratio
model.config.tokenizer_padding_side = llm_tokenizer.padding_side
model.config.tokenizer_model_max_length = llm_tokenizer.model_max_length
model.config.vision_select_feature = model_args.mm_vision_select_feature
model.config.text_select_feature = model_args.text_select_feature
model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
model.config.mm_projector_lr = training_args.mm_projector_lr
model.config.select_layer = model_args.mm_vision_select_layer
training_args.use_im_start_end = model_args.mm_use_im_start_end
model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token


train_dataset = LazySupervisedDataset(  tokenizer=llm_tokenizer,
                                        data_path=data_args.data_path,
                                        data_args=data_args,
                                        vit_text_tokenizer= vit_tokenizer
                                    )

data_collator = DataCollatorForSupervisedDataset(tokenizer=llm_tokenizer,
                                                 vit_text_tokenizer= vit_tokenizer,
                                                 device= model.device)

trainer = LLaVATrainer( model = model,
                        tokenizer = llm_tokenizer,
                        args = training_args,
                        train_dataset = train_dataset,
                        eval_dataset = None,
                        data_collator = data_collator)

trainer.train()