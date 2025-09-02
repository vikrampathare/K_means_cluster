import os
import random
import math
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import google.generativeai as genai

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure Gemini API
GEMINI_API_KEY = "AIzaSZpcCIsThisTheKEyX-SF4PqoE"
genai.configure(api_key=GEMINI_API_KEY)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def simple_kmeans(pixels, k, max_iterations=20):
    """Simple k-means implementation without external dependencies"""
    # Initialize centroids randomly
    centroids = []
    for _ in range(k):
        centroid = [random.randint(0, 255) for _ in range(3)]
        centroids.append(centroid)
    
    for iteration in range(max_iterations):
        # Assign pixels to closest centroid
        clusters = [[] for _ in range(k)]
        
        for pixel in pixels:
            distances = []
            for centroid in centroids:
                distance = sum((pixel[i] - centroid[i]) ** 2 for i in range(3))
                distances.append(distance)
            
            closest_centroid = distances.index(min(distances))
            clusters[closest_centroid].append(pixel)
        
        # Update centroids
        new_centroids = []
        for i, cluster in enumerate(clusters):
            if cluster:
                new_centroid = [
                    sum(pixel[j] for pixel in cluster) // len(cluster)
                    for j in range(3)
                ]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[i])
        
        # Check for convergence
        if new_centroids == centroids:
            break
        
        centroids = new_centroids
    
    return centroids

def apply_kmeans_clustering(image_path, k=3):
    """Apply k-means clustering to reduce colors in an image"""
    # Open image with PIL
    image = Image.open(image_path)
    image = image.convert('RGB')
    
    # Get image data
    width, height = image.size
    pixels = list(image.getdata())
    
    # Apply simple k-means
    centroids = simple_kmeans(pixels, k)
    
    # Replace each pixel with its closest centroid
    new_pixels = []
    for pixel in pixels:
        distances = []
        for centroid in centroids:
            distance = sum((pixel[i] - centroid[i]) ** 2 for i in range(3))
            distances.append(distance)
        
        closest_centroid = distances.index(min(distances))
        new_pixels.append(tuple(centroids[closest_centroid]))
    
    # Create new image
    new_image = Image.new('RGB', (width, height))
    new_image.putdata(new_pixels)
    
    return new_image

def image_to_base64(pil_image):
    """Convert PIL image to base64 string for web display"""
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    k_value = int(request.form.get('k_value', 3))
    
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            print(f"Processing image: {filepath} with k={k_value}")
            
            # Process the image
            clustered_image = apply_kmeans_clustering(filepath, k=k_value)
            print("K-means clustering completed")
            
            # Convert original and processed images to base64 for display
            original_image = Image.open(filepath).convert('RGB')
            print("Original image loaded")
            
            original_b64 = image_to_base64(original_image)
            print("Original image converted to base64")
            
            processed_b64 = image_to_base64(clustered_image)
            print("Processed image converted to base64")
            
            return render_template('result.html', 
                                 original_image=original_b64,
                                 processed_image=processed_b64,
                                 k_value=k_value,
                                 filename=filename)
        
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('index'))
    
    else:
        flash('Invalid file type. Please upload an image file.')
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/chat', methods=['POST'])
def chat_with_gemini():
    try:
        data = request.get_json()
        question = data.get('question', '')
        image_data = data.get('image_data', '')
        k_value = data.get('k_value', 3)
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create context about the image processing
        context = f"""
        You are analyzing a cartoonified image that was created using k-means clustering with k={k_value} color clusters.
        
        The original image was processed using an unsupervised learning algorithm (k-means clustering) to reduce the number of colors and create a cartoon-like effect.
        
        Key information about the process:
        - Algorithm used: k-means clustering
        - Number of color clusters (k): {k_value}
        - Effect: Lower k values create more dramatic cartoon effects, higher k values preserve more detail
        - This is unsupervised learning - no training data was needed
        
        Please answer the user's question about this cartoonified image in a helpful and educational way, explaining both the visual aspects and the machine learning concepts involved when relevant.
        """
        
        # If image data is provided, include it in the analysis
        if image_data:
            # Remove the data:image/png;base64, prefix if present
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            
            # Convert base64 to PIL Image for Gemini
            try:
                image_bytes = base64.b64decode(image_data)
                pil_image = Image.open(BytesIO(image_bytes))
                
                # Generate response with image
                response = model.generate_content([context + "\n\nUser question: " + question, pil_image])
                
            except Exception as e:
                # Fallback to text-only if image processing fails
                response = model.generate_content(context + "\n\nUser question: " + question)
        else:
            # Text-only response
            response = model.generate_content(context + "\n\nUser question: " + question)
        
        return jsonify({
            'response': response.text,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error processing chat request: {str(e)}',
            'success': False
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
