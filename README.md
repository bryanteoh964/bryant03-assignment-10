# Assignment 10: Image Search System

Demo Video: [YouTube Link](https://youtu.be/Few2nv2fPng)

## Project Overview
This project implements a simple image search system using CLIP embeddings. Users can search through a database of images using text queries, image queries, or a hybrid approach that combines both. The system returns the top 5 most similar images along with their similarity scores.

## Key Components

### Search Options
1. **Image Query**
   - Upload custom image or use default house.jpg
   - Finds visually similar images
   - Returns top 5 matches with scores

2. **Text Query**
   - Natural language descriptions
   - Semantic search capability
   - Example: "snowy" finds winter scenes

3. **Hybrid Search**
   - Combines image and text queries
   - Adjustable weight parameter (0.0 to 1.0)
   - Default weight: 0.8 favors text query

### Parameters
- **Query Type**: Choose between image, text, or hybrid search
- **Weight**: Control text vs. image influence (for hybrid)
- **PCA Option**: Use PCA embeddings for image queries

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Place data files in `data/` directory:
   - `coco_images_resized/` folder
   - `image_embeddings.pickle`
   - `house.jpg` (default query image)
3. Start server: `python app.py`
4. Access interface: `http://127.0.0.1:5000`
5. Enter query and view results

## Implementation
- Flask web interface
- CLIP embeddings for image/text understanding
- Cosine similarity ranking
- Real-time image preview
- Support for multiple query types