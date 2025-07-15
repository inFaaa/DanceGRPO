import torch
from PIL import Image

class HPSScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        
        model, _, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            '/home/greenland-user/jinfa/DanceGRPO/data/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin',
            precision='amp', device=device, output_dict=True
        )
        cp = "/home/greenland-user/jinfa/DanceGRPO/data/HPSv2/HPS_v2.1_compressed.pt"
        checkpoint = torch.load(cp, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        processor = get_tokenizer('ViT-H-14')
        
        self.model = model.to(device).eval()
        self.processor = processor
        self.preprocess = preprocess_val
        
    @torch.no_grad()
    def __call__(self, prompts, images):
        scores = []
        for image, caption in zip(images, prompts):
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            text_tensor = self.processor([caption]).to(self.device)
            
            with torch.amp.autocast('cuda'):
                outputs = self.model(image_tensor, text_tensor)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T
                hps_score = torch.diagonal(logits_per_image)
                scores.append(hps_score.cpu().item())
        
        return scores
