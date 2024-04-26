from transformers.configuration_utils import PretrainedConfig
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers import AutoConfig

class ProjectorConfig(PretrainedConfig):
    def __init__(self,
                 projector_type = None,
                 target_hidden_size = None,
                 source_hidden_size = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.projector_type  = projector_type
        self.target_hidden_size = target_hidden_size
        self.source_hidden_size = source_hidden_size
        
class TCVConfig(PretrainedConfig):
    
    model_type = "TCVModel"
    
    def __init__(self,
                 text_config = None, 
                 vision_config = None,
                 projector_config = None,
                 text_model_name = "google-bert/bert-base-uncased",
                 vision_model_name="openai/clip-vit-large-patch14-336",
                 projector_name = "mlp2x_gelu",
                 **kwargs,) -> None:
        
        super().__init__(**kwargs)
        
        if text_config:
            self.text_config = text_config
        else:
            self.text_config = AutoConfig.from_pretrained(text_model_name)
            
        if vision_config:
            self.vision_config = vision_config
        else:
            self.vision_config = CLIPVisionConfig.from_pretrained(vision_model_name)
            self.vision_config._name_or_path = vision_model_name
            
        if projector_config :
            self.projector_config = projector_config
        else:   
            self.projector_config = ProjectorConfig(projector_type = projector_name)
            self.projector_config.source_hidden_size = self.text_config.hidden_size
            self.projector_config.target_hidden_size = self.vision_config.hidden_size
            
        
        self.initializer_factor = 1.0
        
class TCVLlamaConfig(PretrainedConfig):
    model_type = "TCVLlama"
    
    def __init__(self, 
                 llm_config = None,
                 tcv_config = None,
                 vit_to_llm_projector_config = None,
                 llm_model_name = 'meta-llama/Llama-2-7b-chat-hf',
                 vit_to_llm_projector_name = "mlp2x_gelu",
                 tcv_vit_model_name = "openai/clip-vit-large-patch14-336",
                 tcv_text_model_name = "google-bert/bert-base-uncased",
                 tcv_text_to_vit_projector_name = "mlp2x_gelu",
                 **kwargs):
        super().__init__(**kwargs)
        
        if llm_config : 
            self.llm_config =  llm_config
        else:
            self.llm_config =  AutoConfig.from_pretrained(llm_model_name)
            
        if tcv_config: 
            self.tcv_config =  tcv_config
        else:
            self.tcv_config =  TCVConfig(
                text_model_name= tcv_text_model_name,
                vision_model_name= tcv_vit_model_name,
                projector_name=tcv_text_to_vit_projector_name
            )
        
        if vit_to_llm_projector_config:
            self.vit_to_llm_projector_config = vit_to_llm_projector_config
        else:
            self.vit_to_llm_projector_config = ProjectorConfig(projector_type = vit_to_llm_projector_name)
            self.vit_to_llm_projector_config.source_hidden_size = self.tcv_config.vision_config.hidden_size
            self.vit_to_llm_projector_config.target_hidden_size = self.llm_config.hidden_size
        
        self.initializer_factor = 1.0