#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import numpy as np
import faiss

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    return torch.device('cpu')

def load_model(device):
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model.to(device)
    model.eval()
    return model, processor

def encode_images(image_paths, model, processor, device, batch_size=16):
    all_embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        images = [Image.open(p).convert('RGB') for p in batch_paths]

        inputs = processor(images=images, return_tensors='pt', padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            vision_outputs = model.vision_model(**inputs)
            pooled = vision_outputs.pooler_output
            feats = model.visual_projection(pooled)

        feats = feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        all_embeddings.append(feats.cpu().numpy())

    return np.vstack(all_embeddings)


def encode_text(text, model, processor, device):
    inputs = processor(text=[text], return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_outputs = model.text_model(**inputs)
        pooled = text_outputs.pooler_output
        feats = model.text_projection(pooled)

    feats = feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    return feats.cpu().numpy()

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings)
    return index

def search(text_query, model, processor, device, index, paths, top_k=5):
    text_emb = encode_text(text_query, model, processor, device)

    D, I = index.search(text_emb, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        results.append((paths[idx], float(score)))

    return results

def main():
    image_dir = Path('../data/frames')
    device = get_device()

    print(f'Using device: {device}')

    model, processor = load_model(device)

    image_paths = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_paths.extend(list(image_dir.glob(ext)))

    image_paths = sorted(image_paths)

    print(f'Found {len(image_paths)} images')

    embeddings = encode_images(image_paths, model, processor, device)

    print(f'Embeddings shape: {embeddings.shape}')

    index = build_index(embeddings)

    print('FAISS index built')

    while True:
        query = input('\nEnter text query (or "exit"): ')
        if query == 'exit':
            break

        results = search(query, model, processor, device, index, image_paths, top_k=9)

        print('\nTop results:')
        plt.figure(figsize=(10, 10))
        for i, (path, score) in enumerate(results):
            print(f'{score:.4f} - {path}')

            img = Image.open(path).convert('RGB')

            plt.subplot(3, 3, i + 1)
            plt.imshow(img)
            plt.title(f'{score:.2f}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        plt.close()



if __name__ == '__main__':
    main()