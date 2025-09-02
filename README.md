# k-Means Color Clustering Web Application

A Flask web application that demonstrates unsupervised learning through k-means clustering to create "cartoonified" versions of uploaded images.

## Features

- 🖼️ **Image Upload**: Support for PNG, JPG, JPEG, GIF, and BMP formats
- 🎨 **Color Clustering**: Reduce image colors using k-means clustering (2-8 clusters)
- 🎭 **Cartoonification**: Transform photos into cartoon-like images
- 📱 **Responsive Design**: Modern, mobile-friendly interface
- ⬇️ **Download Results**: Save processed images to your device
- 🔄 **Real-time Processing**: Instant image transformation

## How It Works

1. **Upload an Image**: Choose any image from your device
2. **Select k Value**: Choose the number of color clusters (2-8)
3. **k-Means Clustering**: The algorithm groups similar colors together
4. **View Results**: Compare original and cartoonified versions side-by-side

## Technical Details

- **Algorithm**: k-Means clustering using scikit-learn
- **Image Processing**: OpenCV for image manipulation
- **Web Framework**: Flask with Bootstrap 5 UI
- **File Handling**: Secure file uploads with validation

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and navigate to `http://localhost:5000`

## Usage

1. Open the web application in your browser
2. Click "Choose an Image" and select an image file
3. Select the desired number of color clusters (k value)
4. Click "Cartoonify Image" to process
5. View the results and download if desired
6. Try different k values for various effects

## k-Value Effects

- **k=2**: Minimal colors, high contrast cartoon effect
- **k=3**: Simple cartoon with basic color separation
- **k=4-5**: Balanced cartoon effect with moderate detail
- **k=6-8**: More detailed cartoon with richer colors

## File Structure

```
k-means-color-clustering/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Upload page
│   └── result.html       # Results page
├── uploads/              # Uploaded images (created automatically)
└── processed/            # Processed images (created automatically)
```

## Dependencies

- Flask: Web framework
- OpenCV: Image processing
- scikit-learn: k-means clustering algorithm
- NumPy: Numerical operations
- Pillow: Image handling
- Werkzeug: File upload security

## Educational Value

This application demonstrates:
- **Unsupervised Learning**: k-means clustering without labeled data
- **Computer Vision**: Image processing and color space manipulation
- **Web Development**: Full-stack application with Flask
- **User Experience**: Interactive machine learning demonstration

## Troubleshooting

- **Large files**: Maximum upload size is 16MB
- **Unsupported formats**: Only image files are accepted
- **Processing time**: Larger images may take longer to process
- **Memory usage**: Very large images might require more system memory

Enjoy creating cartoon versions of your images with machine learning! 🎨
