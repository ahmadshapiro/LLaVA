from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.tcv.tcv_configs import ProjectorConfig, TCVConfig, TCVLlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling 
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.bert import BertModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import BertConfig, BertModel, CLIPImageProcessor
from transformers.models.clip.modeling_clip import CLIPVisionTransformer, CLIPPreTrainedModel
from transformers import AutoModel, AutoConfig, LlamaForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoConfig, AutoModelForCausalLM
from typing import Optional, Tuple, Union, List
from torch import nn
from peft import get_peft_model
import torch
import re

def build_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.source_hidden_size, config.target_hidden_size) # Shapiro (ViT hidden size, LLama hidden size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.source_hidden_size, config.target_hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.target_hidden_size, config.target_hidden_size))
        return nn.Sequential(*modules)

    raise ValueError(f'Unknown projector type: {projector_type}')

class CLIPTextConditionedVisionModel(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = CLIPTextConditionedVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        text_embeddings: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            text_embeddings=text_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
class CLIPTextConditionedVisionTransformer(CLIPVisionTransformer):
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        text_embeddings: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

    ) -> Union[Tuple, BaseModelOutputWithPooling]:
    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
    
        hidden_states = self.embeddings(pixel_values)
        hidden_states = torch.cat([hidden_states, text_embeddings], dim=1) #TODO: Check the validity
        hidden_states = self.pre_layrnorm(hidden_states)
        
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
                      
class TCVModel(CLIPPreTrainedModel):
    config_class = TCVConfig

    def __init__(self, config: TCVConfig, **kwargs):
        super().__init__(config)
        
        if isinstance(config.text_config, dict):
            self.text_model = BertModel.from_pretrained(BertConfig.from_dict(config.text_config)._name_or_path)
        else:
            self.text_model = BertModel.from_pretrained(config.text_config._name_or_path)
            
        if isinstance(config.projector_config, dict):  
            self.text_projection = build_projector(ProjectorConfig.from_dict(config.projector_config))
        else:
            self.text_projection = build_projector(config.projector_config)
        
        if isinstance(config.vision_config, dict):  
            self.vision_model = CLIPTextConditionedVisionModel.from_pretrained(CLIPVisionConfig.from_dict(config.vision_config)._name_or_path)
            self.image_processor = CLIPImageProcessor.from_pretrained(CLIPVisionConfig.from_dict(config.vision_config)._name_or_path)
        else:
            self.vision_model = CLIPTextConditionedVisionModel.from_pretrained(config.vision_config._name_or_path)
            self.image_processor = CLIPImageProcessor.from_pretrained(config.vision_config._name_or_path)
            
                  
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        text_select_feature : Optional[str] = "all",
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        #TODO : Do we take the pooled outputs from BERT or all tokens ?
        if text_select_feature == "all":
            text_embeddings = self.text_projection(text_outputs['last_hidden_state'])
        else:
            text_embeddings = self.text_projection(text_outputs['pooler_output']).unsqueeze(1)
            
        
        return self.vision_model(
            pixel_values=pixel_values,
            text_embeddings = text_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        
    @property
    def vision_hidden_size(self):
        return self.config.vision_config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.vision_config.image_size // self.config.vision_config.patch_size

    @property
    def num_patches(self):
        return (self.config.vision_config.image_size // self.config.vision_config.patch_size) ** 2
        
# class TCVVisionTower(nn.Module):
#     def __init__(self,
#                  vision_model_name, 
#                  text_model_name,
#                  projector_name,
#                  args,
#                  use_loRa = False,
#                  path = None,
#                  delay_load=False):
        
#         super().__init__()

#         self.is_loaded = False
#         self.path = None
#         self.vision_model_name = vision_model_name
#         self.text_model_name = text_model_name
#         self.projector_name = projector_name
#         self.select_layer = args.mm_vision_select_layer
#         self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

#         if not delay_load:
#             self.load_model()
#         elif getattr(args, 'unfreeze_mm_vision_tower', False):
#             self.load_model()
#         else:
#             self.cfg_only = TCVConfig.from_pretrained(
#                                                         text_model_name = self.text_model_name,
#                                                         vision_model_name = self.vision_model_name,
#                                                         projector_name = self.projector_name)
    
#     def load_model(self, device_map=None):
#         if self.is_loaded:
#             print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_model_name))
#             return

#         self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_model_name)
        
#         if not self.path: 
#             self.vision_tower_config = TCVConfig(text_model_name= self.text_model_name,
#                                                 vision_model_name= self.vision_model_name,
#                                                 projector_name= self.projector_name)
            
#             self.vision_tower = TCVModel(self.vision_tower_config,
#                                          device_map = device_map)
#         else:
#             self.vision_tower = TCVModel.from_pretrained(self.path,
#                                                          device_map = device_map)
            
#             self.vision_tower_config = self.vision_tower.config

#         self.is_loaded = True

#     def make_peft_ViT(self, lora_config,
#                       freeze_text_model = True):
#         self.vision_tower.vision_model = get_peft_model(model = self.vision_tower.vision_model,
#                                                         peft_config = lora_config )
#         if freeze_text_model:
#             for param in self.vision_tower.text_model.parameters():
#                 param.requires_grad = False
        
    
#     def feature_select(self, image_forward_outs):
#         image_features = image_forward_outs.hidden_states[self.select_layer]
#         if self.select_feature == 'patch':
#             image_features = image_features[:, 1:]
#         elif self.select_feature == 'cls_patch':
#             image_features = image_features
#         else:
#             raise ValueError(f'Unexpected select feature: {self.select_feature}')
#         return image_features

#     def forward(self, 
#                 images, 
#                 input_ids, 
#                 attention_mask):
        
#         #TODO : Should be handled here if type is list i don't know when this happens
#         #TODO : Change the model passing to the collator
#         if type(images) is list:
#             image_features = []
#             for image in images:
#                 image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
#                 image_feature = self.feature_select(image_forward_out).to(image.dtype)
#                 image_features.append(image_feature)
#         else:
#             image_forward_outs = self.vision_tower(pixel_values = images.to(device=self.device, dtype=self.dtype), 
#                                                    input_ids = input_ids.to(device=self.device, dtype=self.dtype),
#                                                    attention_mask = attention_mask.to(device=self.device, dtype=self.dtype),
#                                                    output_hidden_states=True)
            
#             image_features = self.feature_select(image_forward_outs).to(images.dtype)

#         return image_features

#     @property
#     def dummy_feature(self):
        
#         return torch.zeros(1, 
#                            self.vision_tower_config.vision_config.hidden_size, 
#                            device=self.device, 
#                            dtype=self.dtype)

#     @property
#     def dtype(self):
#         return self.vision_tower.dtype

#     @property
#     def device(self):
#         return self.vision_tower.device

#     @property
#     def config(self):
#         if self.is_loaded:
#             return self.vision_tower.config
#         else:
#             return self.cfg_only

#     @property
#     def hidden_size(self):
#         return self.vision_tower_config.vision_config.hidden_size

#     @property
#     def num_patches_per_side(self):
#         return self.vision_tower_config.vision_config.image_size // self.vision_tower_config.vision_config.patch_size

#     @property
#     def num_patches(self):
#         return (self.config.image_size // self.config.patch_size) ** 2
        
class TCVLlamaForCausalLM(PreTrainedModel):
    config_class = TCVLlamaConfig
    #What do we need ? 
    ## self.select_feature in self.encode_images we need to pass it from args at the start of the model or add it to the config
    def __init__(self, config : TCVLlamaConfig, **kwargs):
        super().__init__(config,  **kwargs)
        
        if isinstance(config.llm_config, dict):
            self.llm = LlamaForCausalLM.from_pretrained(config.llm_config['_name_or_path'])
        else:
            self.llm = LlamaForCausalLM.from_pretrained(config.llm_config._name_or_path)
            
        if isinstance(config.tcv_config, dict): #TODO : Shoudl we also do from pretrained or not ? 
            self.tcv = TCVModel(TCVConfig.from_dict(config.tcv_config))
        else:
            self.tcv = TCVModel(config.tcv_config)
            
        if isinstance(config.vit_to_llm_projector_config, dict):  
            self.vit_to_llm_projector = build_projector(ProjectorConfig.from_dict(config.vit_to_llm_projector_config))
        else:
            self.vit_to_llm_projector = build_projector(config.vit_to_llm_projector_config)
        
        self.is_peft_wrapped = False

    def unpad_image(tensor, original_size):
        
        original_width, original_height = original_size
        current_height, current_width = tensor.shape[1:]

        original_aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height

        if original_aspect_ratio > current_aspect_ratio:
            scale_factor = current_width / original_width
            new_height = int(original_height * scale_factor)
            padding = (current_height - new_height) // 2
            unpadded_tensor = tensor[:, padding:current_height - padding, :]
        else:
            scale_factor = current_height / original_height
            new_width = int(original_width * scale_factor)
            padding = (current_width - new_width) // 2
            unpadded_tensor = tensor[:, :, padding:current_width - padding]

        return unpadded_tensor
    
    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        vit_text_input_ids: torch.LongTensor = None,
        vit_text_attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids = input_ids.to(self.device),
                position_ids = position_ids,
                attention_mask = attention_mask.to(self.device),
                past_key_values = past_key_values,
                labels = labels,
                images = images.to(self.device),
                vit_text_input_ids = vit_text_input_ids.to(self.device),
                vit_text_attention_mask = vit_text_attention_mask.to(self.device),
                image_sizes = image_sizes
            )

        return self.llm(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds,
                            labels=labels,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict
                        )
        
    def encode_images(self, 
                      images,
                      input_ids,
                      attention_mask):
        
        image_features = self.tcv(
                                    input_ids = input_ids,
                                    attention_mask = attention_mask,
                                    pixel_values = images,
                                    text_select_feature = self.config.text_select_feature
                                  )
        

        image_features = image_features.hidden_states[self.config.select_layer] #TODO Add me to configs
        
        if self.config.vision_select_feature == 'patch': #TODO Add me to configs
            image_features = image_features[:, 1:]
        elif self.config.vision_select_feature == 'cls_patch': #TODO Add me to configs
            image_features = image_features
            
        image_features = self.vit_to_llm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
                                                self, 
                                                input_ids,
                                                position_ids, 
                                                attention_mask, 
                                                past_key_values, 
                                                labels,
                                                images,
                                                vit_text_input_ids,
                                                vit_text_attention_mask,
                                                image_sizes=None
    ):
        
        if self.tcv is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        image_features = self.encode_images(images = images,
                                            input_ids = vit_text_input_ids,
                                            attention_mask = vit_text_attention_mask)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.llm.get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.llm.get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds] #TODO: Check the device here

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None) #TODO Add me to configs
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    
    def wrap_peft(
        self,
        llm_lora_config,
        vit_lora_config,
        freeze_bert_true = True,
    ):
        
        self.llm = get_peft_model(self.llm, llm_lora_config)
        self.tcv.vision_model = get_peft_model(self.tcv.vision_model, vit_lora_config)
        if freeze_bert_true:
            for param in self.tcv.text_model.parameters():
                param.requires_grad = False
        
        self.is_peft_wrapped = True
    
    def get_unwrapped(self):
        llm = self.llm.merge_and_unload()
        vision_model = self.tcv.vision_model.merge_and_unload()
        
        return llm, vision_model

    def save_pretrained(self, **kwargs):
        
        if self.is_peft_wrapped:
            self.llm, self.tcv.vision_model = self.get_unwrapped()
        
        super().save_pretrained(**kwargs)
            
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        vit_text_input_ids = kwargs.pop("vit_text_input_ids", None)
        vit_text_attention_mask = kwargs.pop("vit_text_attention_mask", None)
        
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids = inputs,
                position_ids = position_ids,
                attention_mask = attention_mask,
                past_key_values = None,
                labels = None,
                images = images,
                vit_text_input_ids = vit_text_input_ids.to(self.device),
                vit_text_attention_mask = vit_text_attention_mask.to(self.device),
                image_sizes = image_sizes
            )
        else:
            inputs_embeds = self.llm.get_input_embeddings()(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
    
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features


AutoConfig.register("TCVModel", TCVConfig)
AutoConfig.register("TCVLlama", TCVLlamaConfig)

AutoModel.register(TCVConfig, TCVModel)

AutoModel.register(TCVLlamaConfig, TCVLlamaForCausalLM)
AutoModelForCausalLM.register(TCVLlamaConfig, TCVLlamaForCausalLM)


            