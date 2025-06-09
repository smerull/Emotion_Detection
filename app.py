import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
import string
from transformers import BertTokenizer, TFBertModel
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.regularizers import l2
import os
import pandas as pd

# Configuration
label_cols = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']
max_len = 128

# Custom CSS to match the HTML template style
def load_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main container */
    .main .block-container {
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        padding: 0;
        max-width: 800px;
        margin-top: 20px;
    }
    
    /* Header styling */
    .custom-header {
        background: linear-gradient(135deg, #ff6b6b, #ffd93d);
        padding: 30px;
        text-align: center;
        color: white;
        border-radius: 20px 20px 0 0;
        margin: -1rem -1rem 2rem -1rem;
    }
    
    .custom-header h1 {
        font-size: 2.5rem;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        color: white;
    }
    
    .custom-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        color: white;
        margin: 0;
    }
    
    /* Hide default streamlit header */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    /* Emotion cards */
    .emotion-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #dee2e6;
        transition: all 0.3s ease;
    }
    
    .emotion-card.detected {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left-color: #28a745;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.2);
    }
    
    /* Custom progress bar */
    .custom-progress {
        background: #e9ecef;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin-top: 8px;
        width: 100%;
    }
    
    .custom-progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .custom-progress-fill.detected {
        background: linear-gradient(90deg, #28a745, #20c997);
    }
    
    /* Summary box */
    .summary-box {
        background: linear-gradient(135deg, #fff3cd, #ffeeba);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #ffc107;
        color: #856404;
    }
    
    /* Example texts */
    .example-item {
        background: white;
        padding: 10px 15px;
        margin-bottom: 8px;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid #e9ecef;
    }
    
    .example-item:hover {
        background: #e9ecef;
        transform: translateX(5px);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 15px 20px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Text area */
    .stTextArea textarea {
        border: 2px solid #e1e5e9;
        border-radius: 10px;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Metrics */
    .metric-container {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
    }
    
    /* Hide hamburger menu */
    .css-14xtw13.e8zbici0 {
        display: none;
    }
    
    /* Hide "Made with Streamlit" */
    footer {
        display: none;
    }
    
    .css-h5rgaw.egzxvld1 {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

# Page config
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
load_custom_css()

def create_multilabel_model(input_shape, num_labels):
    """Recreate the exact same model architecture from training"""
    inputs = Input(shape=input_shape)
    x = Dropout(0.2)(inputs)

    x = LSTM(
        128,
        return_sequences=True,
        dropout=0.3,
        recurrent_dropout=0.3,
        kernel_regularizer=l2(1e-5),
        recurrent_regularizer=l2(1e-5),
        bias_regularizer=l2(1e-5)
    )(x)

    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_labels, activation='sigmoid')(x)
    
    return Model(inputs, outputs)

@st.cache_resource
def load_model_components():
    """Load all model components with caching"""
    try:
        # GPU setup
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        
        # Load BERT components
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        bert_model.trainable = False
        
        # Create model architecture
        input_shape = (max_len, 768)
        num_labels = len(label_cols)
        model = create_multilabel_model(input_shape, num_labels)
        
        # Compile model
        model.compile(
            optimizer=Adam(1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Load weights
        weights_path = 'save_model/lstm_model.h5'
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
        else:
            return None, None, None, None
        
        # Load thresholds
        try:
            with open('save_model/thresholds.pkl', 'rb') as f:
                optimal_thresholds = pickle.load(f)
        except FileNotFoundError:
            optimal_thresholds = {
                'Anger': 0.45, 'Fear': 0.35, 'Joy': 0.55,
                'Sadness': 0.40, 'Surprise': 0.50
            }
        
        return model, bert_tokenizer, bert_model, optimal_thresholds
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def preprocess_text(text):
    """Preprocessing teks sama persis dengan training"""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_with_bert(texts, bert_tokenizer, bert_model, max_length=128):
    """Generate BERT embedding untuk teks"""
    if isinstance(texts, str):
        texts = [texts]
    
    encoded = bert_tokenizer(
        texts, padding='max_length', truncation=True,
        max_length=max_length, return_tensors='tf'
    )
    outputs = bert_model(encoded)
    return outputs.last_hidden_state

def predict_emotions(text, model, bert_tokenizer, bert_model, optimal_thresholds):
    """Predict emotions for given text"""
    try:
        processed_text = preprocess_text(text)
        embedding = process_with_bert([processed_text], bert_tokenizer, bert_model, max_length=max_len)
        predictions = model.predict(embedding, verbose=0)[0]
        
        results = {}
        detected_emotions = []
        
        for i, emotion in enumerate(label_cols):
            probability = float(predictions[i])
            threshold = float(optimal_thresholds[emotion])
            is_predicted = bool(probability >= threshold)
            
            results[emotion] = {
                'probability': round(probability, 4),
                'percentage': round(probability * 100, 1),
                'predicted': is_predicted,
                'threshold': round(threshold, 3)
            }
            
            if is_predicted:
                detected_emotions.append({
                    'emotion': emotion,
                    'confidence': round(probability * 100, 1)
                })
        
        detected_emotions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'original_text': text,
            'processed_text': processed_text,
            'detected_emotions': detected_emotions,
            'results': results,
            'predictions': predictions
        }
        
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

def display_emotion_cards(results):
    """Display emotion cards with custom styling"""
    html_content = ""
    
    # Summary box if emotions detected
    if results['detected_emotions']:
        top_emotion = results['detected_emotions'][0]
        html_content += f"""
        <div class="summary-box">
            <h3>🎯 Primary Emotion: {top_emotion['emotion']}</h3>
            <p><strong>Confidence:</strong> {top_emotion['confidence']}%</p>
            <p><strong>Total Detected:</strong> {len(results['detected_emotions'])} emotion(s)</p>
        </div>
        """
    
    # Individual emotion cards
    for emotion in label_cols:
        result = results['results'][emotion]
        is_detected = result['predicted']
        detected_class = 'detected' if is_detected else ''
        badge_text = 'DETECTED' if is_detected else 'Not Detected'
        
        html_content += f"""
        <div class="emotion-card {detected_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 1.1rem; color: #333;">{emotion}</span>
                <span style="background: {'#28a745' if is_detected else '#6c757d'}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                    {badge_text}
                </span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                <span>Confidence: {result['percentage']}%</span>
                <span style="font-size: 0.9rem; color: #6c757d;">Threshold: {int(result['threshold'] * 100)}%</span>
            </div>
            <div class="custom-progress">
                <div class="custom-progress-fill {detected_class}" style="width: {result['percentage']}%"></div>
            </div>
        </div>
        """
    
    st.markdown(html_content, unsafe_allow_html=True)

def display_examples():
    """Display example texts"""
    examples = [
        "I am so happy and excited about this news!",
        "This situation makes me really angry and frustrated!",
        "I feel scared and worried about what might happen.",
        "What a surprising and unexpected turn of events!",
        "I am feeling very sad and lonely today."
    ]
    
    st.markdown("### 💡 Try these examples:")
    
    for example in examples:
        if st.button(f'"{example}"', key=f"example_{hash(example)}", use_container_width=True):
            st.session_state.example_text = example

def main():
    # Custom header
    st.markdown("""
    <div class="custom-header">
        <h1>🎭 Emotion Detection</h1>
        <p>Analyze emotions in text using BERT + LSTM</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model components
    model, bert_tokenizer, bert_model, optimal_thresholds = load_model_components()
    
    if model is None:
        st.error("❌ Model gagal dimuat. Pastikan file model tersedia di folder save_model/")
        st.stop()
    
    # Initialize session state
    if 'example_text' not in st.session_state:
        st.session_state.example_text = ""
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        st.markdown("**Enter your text:**")
        text_input = st.text_area(
            "",
            value=st.session_state.example_text,
            placeholder="Type your text here... (e.g., 'I am so happy today!' or 'This makes me really angry')",
            height=120,
            key="main_text_input"
        )
        
        # Clear example text after use
        if st.session_state.example_text:
            st.session_state.example_text = ""
        
        # Buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            analyze_btn = st.button("🔍 Analyze Emotions", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_btn:
            st.rerun()
        
        # Analysis results
        if analyze_btn and text_input.strip():
            with st.spinner("Analyzing emotions..."):
                result = predict_emotions(text_input, model, bert_tokenizer, bert_model, optimal_thresholds)
                
                if result:
                    st.markdown("## 📊 Analysis Results")
                    display_emotion_cards(result)
                    
                    # Show processed text details
                    with st.expander("🔍 Text Processing Details"):
                        st.write(f"**Original:** {result['original_text']}")
                        st.write(f"**Processed:** {result['processed_text']}")
        
        elif analyze_btn and not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze!")
    
    with col2:
        # Model info
        st.markdown("### 📊 Model Info")
        st.info(f"""
        **Emotions:** {', '.join(label_cols)}
        
        **Architecture:** BERT + LSTM
        
        **Max Length:** {max_len}
        
        **TensorFlow:** {tf.__version__}
        """)
        
        # Thresholds
        st.markdown("### 🎯 Thresholds")
        for emotion, threshold in optimal_thresholds.items():
            st.markdown(f"**{emotion}:** {threshold}")
        
        # Examples
        display_examples()

if __name__ == "__main__":
    main()