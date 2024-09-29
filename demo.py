import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# print(model)
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)  # 将单词编码成序号（本质是查一个固定的map字典）

with torch.no_grad():
    image_features = model.encode_image(image)  #  [1, 512]
    text_features = model.encode_text(text)  #  [3, 512]
    
    logits_per_image, logits_per_text = model(image, text)  # [1, 3]  [3, 1]
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()  # [1, 3]

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]].