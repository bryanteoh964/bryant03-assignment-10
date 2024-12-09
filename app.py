from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import open_clip
import os

app = Flask(__name__)

# Load model and transforms
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# Load embeddings
df = pd.read_pickle('data/image_embeddings.pickle')
embeddings_tensor = torch.tensor(df['embedding'].tolist())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)

@app.route('/search', methods=['POST'])
def search():
    text_query = request.form.get('text_query', '')
    image_file = request.files.get('image')
    weight = float(request.form.get('weight', 0.8))  # Default to 0.8 as shown in example
    query_type = request.form.get('query_type', 'hybrid')
    
    # Process text query if provided and if not image-only query
    if text_query and query_type != 'image':
        text = tokenizer([text_query])
        text_query_embedding = F.normalize(model.encode_text(text))
    else:
        text_query_embedding = None
    
    # Process image query if provided and if not text-only query
    if query_type != 'text':
        image_path = 'data/house.jpg'
        if image_file:
            image_file.save('temp_image.jpg')
            image_path = 'temp_image.jpg'
        
        image = preprocess(Image.open(image_path)).unsqueeze(0)
        image_query_embedding = F.normalize(model.encode_image(image))
        
        if image_file:
            os.remove('temp_image.jpg')
    else:
        image_query_embedding = None
    
    # Determine final query based on query type
    if query_type == 'text':
        query = text_query_embedding
    elif query_type == 'image':
        query = image_query_embedding
    else:  # hybrid
        if text_query_embedding is not None and image_query_embedding is not None:
            query = F.normalize(weight * text_query_embedding + (1.0 - weight) * image_query_embedding)
        elif text_query_embedding is not None:
            query = text_query_embedding
        else:
            query = image_query_embedding
    
    # Get top 5 results
    similarities = F.cosine_similarity(query, embeddings_tensor)
    top_indices = similarities.argsort(descending=True)[:5].tolist()  # Convert tensor to list
    
    results = []
    for idx in top_indices:
        similarity = similarities[idx].item()
        image_path = os.path.join('data/coco_images_resized/coco_images_resized', df.iloc[int(idx)]['file_name'])  # Convert to int
        results.append({
            'image_path': image_path,
            'similarity': f"{similarity:.3f}"
        })
    
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)