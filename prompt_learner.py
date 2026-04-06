import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import Optional, List


class _CLIPTextEncoder(nn.Module):
    """Stable CLIP text encoder for prompt embeddings."""
    def __init__(self, clip_model, dtype=torch.float32):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = dtype
        self.context_length = clip_model.context_length

    def forward(self, prompt_embeds: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = prompt_embeds.to(self.dtype)
        
        pos_emb = self.positional_embedding.to(self.dtype)
        if pos_emb.shape[0] < x.shape[1]:
            pos_emb = F.pad(pos_emb, (0, 0, 0, x.shape[1] - pos_emb.shape[0]))
        x = x + pos_emb.unsqueeze(0)

        x = x.permute(1, 0, 2)  
        x = self.transformer(x)
        x = x.permute(1, 0, 2) 
        x = self.ln_final(x)

        eot_pos = tokenized_prompts.argmax(dim=-1)  
        x = x[torch.arange(x.size(0), device=x.device), eot_pos]
        
        if self.text_projection is not None:
            x = x @ self.text_projection.to(self.dtype)
            
        return x


class PromptLearner(nn.Module):
    def __init__(
        self, 
        classnames: List[str], 
        clip_model, 
        n_ctx: int = 16, 
        ctx_init: Optional[str] = None,
        device: str = "cuda", 
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.device = device
        self.dtype = dtype
        self.n_classes = len(classnames)
        self.context_length = clip_model.context_length

        ctx_dim = clip_model.ln_final.weight.shape[0] 
        vis_dim = clip_model.visual.output_dim 

        self.meta_net = nn.Sequential(
            nn.Linear(vis_dim, vis_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(vis_dim // 16, ctx_dim),
        ).to(dtype)

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            init_tokens = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                init_embed = clip_model.token_embedding(init_tokens).type(dtype)
            ctx = init_embed[0, 1:1 + n_ctx, :].clone()
            
            if ctx.shape[0] < n_ctx:
                pad = torch.zeros(n_ctx - ctx.shape[0], ctx_dim, dtype=dtype, device=device)
                ctx = torch.cat([ctx, pad], dim=0)
        else:
            ctx = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device)
            nn.init.normal_(ctx, std=0.02)

        self.ctx = nn.Parameter(ctx) 

        classnames = [c.replace("_", " ") for c in classnames]
        prompt_prefix = " ".join(["X"] * n_ctx)
        
        prompts = []
        for name in classnames:
            name = name.strip()
            if not name.endswith('.'):
                name = name + '.'
            prompts.append(f"{prompt_prefix} {name}")
        
        tokenized_all = []
        for p in prompts:
            tokens = clip.tokenize(p).to(device)
            if tokens.shape[1] > self.context_length:
                tokens = tokens[:, :self.context_length]
            tokenized_all.append(tokens)
        
        tokenized_all = torch.cat(tokenized_all, dim=0)
        self.register_buffer("tokenized_prompts_all", tokenized_all)

        with torch.no_grad():
            emb = clip_model.token_embedding(tokenized_all).type(dtype) 

        self.register_buffer("token_prefix", emb[:, :1, :]) 
        
        suffix_start = 1 + n_ctx
        self.register_buffer("token_suffix", emb[:, suffix_start:, :])

        print(f"Initialized PromptLearner with {len(classnames)} classes, n_ctx={n_ctx}")

    def forward(
        self, 
        image_features: torch.Tensor, 
        target_labels: torch.Tensor
    ) -> torch.Tensor:
        
        B = image_features.shape[0]
        image_features = image_features.to(self.device).to(self.dtype)
        
        bias = self.meta_net(image_features) 
        
        ctx = self.ctx.unsqueeze(0).expand(B, -1, -1)  
        ctx_shifted = ctx + bias.unsqueeze(1) 

        prefix = self.token_prefix[target_labels]  
        suffix = self.token_suffix[target_labels] 
        
        prompt_embeds = torch.cat([prefix, ctx_shifted, suffix], dim=1)
        
        return prompt_embeds


class Conditioner(nn.Module):

    def __init__(
        self, 
        classnames: List[str], 
        clip_backbone: str = "ViT-B/16", 
        device: str = "cuda", 
        n_ctx: int = 16, 
        ctx_init: Optional[str] = None
    ):
        super().__init__()
        
        clip_model, _ = clip.load(clip_backbone, device=device, jit=False)
        
        clip_model = clip_model.float()
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)

        self._clip = clip_model
        self.device = device
        self.dtype = torch.float32
        self.n_ctx = n_ctx
        
        self.prompt_learner = PromptLearner(
            classnames=classnames,
            clip_model=clip_model,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            device=device,
            dtype=self.dtype,
        )
    
        self.text_encoder = _CLIPTextEncoder(clip_model, dtype=self.dtype)
        
        self.clip_res = clip_model.visual.input_resolution
        self.normalize_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=self.dtype)
        self.normalize_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=self.dtype)
        self.to(device)
        
    def _preprocess_for_clip(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess images for CLIP encoder."""
        x = x.to(self.dtype)

        if x.shape[-2] != self.clip_res or x.shape[-1] != self.clip_res:
            x = F.interpolate(
                x, 
                size=(self.clip_res, self.clip_res), 
                mode='bilinear', 
                align_corners=False
            )

        mean = self.normalize_mean.view(1, 3, 1, 1).to(x.device)
        std = self.normalize_std.view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        
        return x
    
    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        x = self._preprocess_for_clip(images)
        img_feat = self._clip.encode_image(x)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        return img_feat
    
    def forward(
        self, 
        images: torch.Tensor, 
        target_local: torch.Tensor
    ) -> torch.Tensor:

        images = images.to(self.device)
        target_local = target_local.to(self.device)
        
        img_feat = self.encode_image(images)  

        prompt_embeds = self.prompt_learner(img_feat, target_local) 
        
        tokenized = self.prompt_learner.tokenized_prompts_all[target_local]  

        text_feat = self.text_encoder(prompt_embeds, tokenized)  
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            
        return text_feat
    
    def train(self, mode: bool = True):
        """Set training mode. Only prompt_learner is trainable."""
        super().train(mode)
        self._clip.eval()
        for p in self._clip.parameters():
            p.requires_grad_(False)
        return self
