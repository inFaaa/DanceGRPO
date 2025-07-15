import torch
import torch.nn.functional as F
from PIL import Image

class CLIPScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        from open_clip import create_model_from_pretrained, get_tokenizer
        
        processor = get_tokenizer('ViT-H-14')
        model, preprocess = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384')
        
        self.model = model.to(device).eval()
        self.processor = processor
        self.preprocess = preprocess
        
    @torch.no_grad()
    def __call__(self, prompts, images):
        scores = []
        for image, caption in zip(images, prompts):
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            text_tensor = self.processor([caption], context_length=self.model.context_length).to(self.device)
            
            with torch.amp.autocast('cuda'):
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tensor)
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                clip_score = image_features @ text_features.T
                scores.append(clip_score.squeeze(0).cpu().item())
        
        return scores
