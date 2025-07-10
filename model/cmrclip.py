import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from model.video_transformer import SpaceTimeTransformer
from utils.util import state_dict_data_parallel_fix
from transformers import AutoModel

class CMRCLIP(BaseModel):
    def __init__(self,
                 video_params,
                 text_params,
                 projection_dim=512,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='zeros'):
        super().__init__()

        self.video_params = video_params
        self.text_params = text_params
        self.load_temporal_fix = load_temporal_fix
        self.projection_dim = projection_dim
        
        # Initialize text encoder
        self.text_model = AutoModel.from_pretrained(text_params['model'])
        self.text_model.train()
        
        # Initialize video encoder        
        self.video_model = self._init_video_encoder(video_params)
        
        # Projection layers
        self.txt_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.text_model.config.hidden_size, projection_dim)
        )
        
        self.vid_proj = nn.Sequential(
            nn.ReLU(),  
            nn.Linear(self.video_model.embed_dim, projection_dim)
        )
    
        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint)
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            self.load_state_dict(new_state_dict, strict=False)

    def _init_video_encoder(self, video_params):
        """Initialize the video encoder with SpaceTimeTransformer."""
        model = SpaceTimeTransformer(
            num_frames=video_params['num_frames'],
            time_init=video_params['time_init'],
            attention_style='frozen-in-time'
        )
    
        model.head = nn.Identity()
        model.pre_logits = nn.Identity()

        vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=video_params['pretrained'])
        vit_checkpoint = vit_model.state_dict()
        model.load_state_dict(vit_checkpoint, strict=False)
        
        return model

    def set_device(self, device):
        self.device = device

    def forward(self, data, return_embeds=True):
        text_data = data['text']
        video_data = data['video']
        
        text_embeddings = self.compute_text(text_data)
        video_embeddings = self.compute_video(video_data)
        
        if return_embeds:
            return text_embeddings, video_embeddings

        return sim_matrix(text_embeddings, video_embeddings)

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['video_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['video_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['video_model.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['video_model.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict

    
    
    def compute_text(self, text_data):
        """Extract text embeddings based on the model type."""
        model_name = self.text_params['model'].lower()
        max_len = 512
            
        input_ids = text_data['input_ids'][:, :max_len]
        attention_mask = text_data['attention_mask'][:, :max_len]
        
        outputs = self.text_model(input_ids, attention_mask=attention_mask)
        
        if model_name.startswith('distilbert'):
            text_features = outputs.last_hidden_state[:, 0, :]
        elif 'bert' in model_name:
            # Mean pooling of last hidden state
            last_hidden_state = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            text_features = sum_embeddings / sum_mask
        else:
            text_features = outputs['pooler_output']

        # Apply projection
        text_embeddings = self.txt_proj(text_features.float())
        return text_embeddings

    def compute_video(self, video_data):
        """Extract video embeddings."""
        video_features = self.video_model(video_data)
        video_embeddings = self.vid_proj(video_features)
        return video_embeddings

def sim_matrix(a, b, eps=1e-8, chunk_size=1024):
    """
    added eps for numerical stability
    optimized with chunking for memory efficiency
    """
    if a.shape[0] <= chunk_size and b.shape[0] <= chunk_size:
        # If both matrices are small enough, compute directly
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        
        # Convert half precision tensors to float before matrix multiplication
        if a_norm.dtype == torch.float16 or b_norm.dtype == torch.float16:
            a_norm = a_norm.float()  # Convert to float32
            b_norm = b_norm.float()  # Convert to float32
        
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt
    else:
        # Process in chunks to save memory
        sim_mt = torch.zeros(a.shape[0], b.shape[0], device=a.device)
        
        # Normalize a
        a_n = a.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        
        # Normalize b
        b_n = b.norm(dim=1)[:, None]
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        
        # Convert half precision tensors to float before matrix multiplication
        if a_norm.dtype == torch.float16 or b_norm.dtype == torch.float16:
            a_norm = a_norm.float()  # Convert to float32
            b_norm = b_norm.float()  # Convert to float32
        
        # Compute similarity matrix in chunks
        for i in range(0, a.shape[0], chunk_size):
            end_i = min(i + chunk_size, a.shape[0])
            for j in range(0, b.shape[0], chunk_size):
                end_j = min(j + chunk_size, b.shape[0])
                sim_mt[i:end_i, j:end_j] = torch.mm(a_norm[i:end_i], b_norm[j:end_j].transpose(0, 1))
                
        return sim_mt
    
    def compute_similarity(a, b, a_mask=None, b_mask=None, style='single', eps=1e-8, return_raw=False, temp=0.5):
        if style == 'single':
            sim = sim_matrix(a, b, eps=eps)
            return sim, sim.t()
        else:
            raise NotImplementedError

if __name__ == "__main__":
    pass
