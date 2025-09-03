import gradio as gr
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import warnings

warnings.filterwarnings('ignore')

# Load your trained model and tokenizer
model_path = r"saved_model"
tokenizer_path = r"saved_model"

try:
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    print("Model and tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to pre-trained model for demonstration
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    print("Using fallback pre-trained model")


def clean_text(text):
    """Clean and preprocess text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    return text.strip()


def scrape_article_content(url):
    """Scrape article content from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Try to find article content using common selectors
        article_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.content',
            '[class*="article"]',
            '[class*="post"]',
            'main'
        ]

        article_text = ""
        for selector in article_selectors:
            elements = soup.select(selector)
            if elements:
                article_text = ' '.join([elem.get_text() for elem in elements])
                break

        # Fallback: get all paragraph text
        if not article_text:
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text() for p in paragraphs])

        # Clean the extracted text
        article_text = clean_text(article_text)

        # Get title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()

        if len(article_text) < 100:
            return None, "Could not extract sufficient article content from the URL"

        return f"Title: {title}\n\nContent: {article_text}", None

    except requests.exceptions.RequestException as e:
        return None, f"Error fetching URL: {str(e)}"
    except Exception as e:
        return None, f"Error processing content: {str(e)}"


def model_predict(text, model, tokenizer):
    """Predict if news is fake or real using your DistilBERT model"""
    try:
        # Tokenize the input text
        encodings = tokenizer(
            [text],
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        # Extract results
        prediction = preds[0].item()
        confidence_scores = probs[0].tolist()

        # Map prediction to label (assuming 1 = True/Real, 0 = Fake)
        label = "Real" if prediction == 1 else "Fake"
        confidence = confidence_scores[prediction] * 100

        return label, confidence, confidence_scores

    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0, [0.0, 0.0]


def analyze_text_input(text):
    """Analyze direct text input"""
    if not text or len(text.strip()) < 10:
        return "Please enter sufficient text for analysis (at least 10 characters)", "", {}

    cleaned_text = clean_text(text)
    label, confidence, scores = model_predict(cleaned_text, model, tokenizer)

    # Format prediction result
    if label == "Fake":
        prediction_result = f"The news appears to be Fake!!"
        result_color = "#dc3545"
    else:
        prediction_result = f" News Appears to be authentic"
        result_color = "#28a745"

    confidence_text = f"Confidence: {confidence:.1f}%"

    # Create detailed breakdown
    breakdown = {
        "Fake News Probability": f"{scores[0]:.1%}",
        "Real News Probability": f"{scores[1]:.1%}",
        "Final Classification": label,
        "Model Confidence": f"{confidence:.1f}%"
    }

    return prediction_result, confidence_text, breakdown


def analyze_url_input(url):
    """Analyze content from URL"""
    if not url or not url.strip():
        return "Please enter a valid URL", "", {}, ""

    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    # Extract content from URL
    content, error = scrape_article_content(url)

    if error:
        return f"❌ Error: {error}", "", {}, ""

    if not content:
        return "❌ Could not extract content from URL", "", {}, ""

    # Analyze the extracted content
    # Extract just the article content (remove title for analysis)
    content_lines = content.split('\n')
    article_content = '\n'.join(content_lines[2:]) if len(content_lines) > 2 else content

    label, confidence, scores = model_predict(article_content, model, tokenizer)

    # Format prediction result
    if label == "Fake":
        prediction_result = f"🚨 FAKE NEWS DETECTED"
    else:
        prediction_result = f"✅ NEWS APPEARS AUTHENTIC"

    confidence_text = f"Confidence: {confidence:.1f}%"

    # Create detailed breakdown
    breakdown = {
        "Fake News Probability": f"{scores[0]:.1%}",
        "Real News Probability": f"{scores[1]:.1%}",
        "Final Classification": label,
        "Model Confidence": f"{confidence:.1f}%"
    }

    # Limit content preview
    content_preview = content[:2000] + "..." if len(content) > 2000 else content

    return prediction_result, confidence_text, breakdown, content_preview


# Custom CSS for professional styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    max-width: 1400px;
    margin: 0 auto;
}

.header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    margin-bottom: 2rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.warning-text {
    color: #856404;
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 6px;
    padding: 1rem;
    margin: 1rem 0;
    font-size: 0.95em;
}

.tab-nav {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 0.5rem;
}

/* Custom button styling */
.primary-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    color: white;
    font-weight: 600;
    transition: all 0.3s ease;
}

.primary-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="AI Fake News Detector") as app:
    gr.HTML("""
    <div class="header">
        <h1>🔍 AI Fake News Detection System</h1>
        <p>Advanced DistilBERT-powered analysis to detect misinformation and verify news authenticity</p>
    </div>
    """)

    gr.HTML("""
    <div class="warning-text">
        <strong>⚠️ Important Notice:</strong> This AI tool provides analysis based on linguistic patterns and should be used as one factor in news verification. 
        Always cross-reference with multiple reliable sources and apply critical thinking when evaluating news content.
    </div>
    """)

    with gr.Tabs() as tabs:

        # Text Analysis Tab
        with gr.TabItem("📝 Direct Text Analysis", id="text_tab"):
            gr.Markdown("### Analyze News Article Text")
            gr.Markdown("*Paste the complete news article text below for analysis*")

            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="News Article Text",
                        placeholder="Enter the complete news article text here...\n\nFor best results, include the full article content including headline and body text.",
                        lines=10,
                        max_lines=20,
                        info="Minimum 10 characters required"
                    )

                    text_analyze_btn = gr.Button(
                        "🔍 Analyze Article Text",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-button"]
                    )

                with gr.Column(scale=2):
                    text_prediction = gr.Textbox(
                        label="Analysis Result",
                        interactive=False,
                        lines=2
                    )

                    text_confidence = gr.Textbox(
                        label="Model Confidence",
                        interactive=False,
                        lines=1
                    )

                    text_breakdown = gr.JSON(
                        label="Detailed Probability Breakdown",
                        show_label=True
                    )

        # URL Analysis Tab
        with gr.TabItem("🌐 URL Content Analysis", id="url_tab"):
            gr.Markdown("### Analyze News from URL")
            gr.Markdown("*Enter a news article URL to automatically extract and analyze content*")

            with gr.Row():
                with gr.Column(scale=3):
                    url_input = gr.Textbox(
                        label="News Article URL",
                        placeholder="https://example-news-site.com/article-title",
                        lines=1,
                        info="Enter the full URL of the news article"
                    )

                    url_analyze_btn = gr.Button(
                        "🔍 Extract & Analyze URL",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-button"]
                    )

                    with gr.Accordion("📄 Extracted Content Preview", open=False):
                        extracted_content = gr.Textbox(
                            label="Article Content",
                            lines=8,
                            max_lines=15,
                            interactive=False,
                            show_label=False
                        )

                with gr.Column(scale=2):
                    url_prediction = gr.Textbox(
                        label="Analysis Result",
                        interactive=False,
                        lines=2
                    )

                    url_confidence = gr.Textbox(
                        label="Model Confidence",
                        interactive=False,
                        lines=1
                    )

                    url_breakdown = gr.JSON(
                        label="Detailed Probability Breakdown",
                        show_label=True
                    )

    # Information and Tips Section
    with gr.Accordion("ℹ️ How to Use & Model Information", open=False):
        gr.Markdown("""
        ### How to Use This Tool

        **Text Analysis:**
        - Copy and paste the complete news article text
        - Include headlines and full article content for best results
        - Minimum 10 characters required

        **URL Analysis:**
        - Enter the full URL of the news article
        - The system will automatically extract the article content
        - Works with most major news websites

        ### Model Information
        - **Architecture**: DistilBERT for Sequence Classification
        - **Classes**: Binary classification (Real vs Fake)
        - **Max Length**: 256 tokens
        - **Output**: Probability scores for both classes

        ### Interpretation Guide
        - **Confidence > 80%**: High confidence prediction
        - **Confidence 60-80%**: Moderate confidence
        - **Confidence < 60%**: Low confidence - verify with additional sources

        ### Best Practices
        - Use complete articles rather than excerpts
        - Cross-reference with multiple sources
        - Consider the publication source and date
        - Apply critical thinking alongside AI analysis
        """)


    # Event handlers
    def handle_text_analysis(text):
        """Handle text input analysis"""
        try:
            prediction, confidence, breakdown = analyze_text_input(text)
            return prediction, confidence, breakdown
        except Exception as e:
            return f"Error during analysis: {str(e)}", "", {}


    def handle_url_analysis(url):
        """Handle URL input analysis"""
        try:
            prediction, confidence, breakdown, content = analyze_url_input(url)
            return prediction, confidence, breakdown, content
        except Exception as e:
            return f"Error during URL analysis: {str(e)}", "", {}, ""


    # Connect event handlers
    text_analyze_btn.click(
        fn=handle_text_analysis,
        inputs=[text_input],
        outputs=[text_prediction, text_confidence, text_breakdown]
    )

    url_analyze_btn.click(
        fn=handle_url_analysis,
        inputs=[url_input],
        outputs=[url_prediction, url_confidence, url_breakdown, extracted_content]
    )

    # Allow Enter key to trigger analysis
    text_input.submit(
        fn=handle_text_analysis,
        inputs=[text_input],
        outputs=[text_prediction, text_confidence, text_breakdown]
    )

    url_input.submit(
        fn=handle_url_analysis,
        inputs=[url_input],
        outputs=[url_prediction, url_confidence, url_breakdown, extracted_content]
    )


def model_predict(text, model, tokenizer):
    """Predict if news is fake or real using DistilBERT model"""
    try:
        # Tokenize the input text
        encodings = tokenizer(
            [text],
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        # Extract results
        prediction = preds[0].item()
        confidence_scores = probs[0].tolist()

        # Map prediction to label (1 = Real/True, 0 = Fake)
        label = "Real" if prediction == 1 else "Fake"
        confidence = confidence_scores[prediction] * 100

        return label, confidence, confidence_scores

    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0, [0.0, 0.0]


def analyze_text_input(text):
    """Analyze direct text input"""
    if not text or len(text.strip()) < 10:
        return "Please enter sufficient text for analysis (at least 10 characters)", "", {}

    cleaned_text = clean_text(text)
    label, confidence, scores = model_predict(cleaned_text, model, tokenizer)

    if label == "Error":
        return "❌ Error during analysis", "", {}

    # Format prediction result
    if label == "Fake":
        prediction_result = f"🚨 FAKE NEWS DETECTED"
    else:
        prediction_result = f"✅ NEWS APPEARS AUTHENTIC"

    confidence_text = f"{confidence:.1f}%"

    # Create detailed breakdown
    breakdown = {
        "Fake News Probability": f"{scores[0]:.2%}",
        "Real News Probability": f"{scores[1]:.2%}",
        "Final Classification": label,
        "Model Confidence": f"{confidence:.1f}%",
        "Prediction Index": int(1 if label == "Real" else 0)
    }

    return prediction_result, confidence_text, breakdown


def analyze_url_input(url):
    """Analyze content from URL"""
    if not url or not url.strip():
        return "Please enter a valid URL", "", {}, ""

    # Add protocol if missing
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    # Extract content from URL
    content, error = scrape_article_content(url)

    if error:
        return f"❌ {error}", "", {}, ""

    if not content:
        return "❌ Could not extract sufficient content from URL", "", {}, ""

    # Analyze the extracted content
    # Extract just the article content (remove title for analysis if present)
    content_lines = content.split('\n')
    if content.startswith('Title:') and len(content_lines) > 2:
        article_content = '\n'.join(content_lines[2:]).strip()
        if article_content.startswith('Content:'):
            article_content = article_content[8:].strip()
    else:
        article_content = content

    label, confidence, scores = model_predict(article_content, model, tokenizer)

    if label == "Error":
        return "❌ Error during content analysis", "", {}, content[:1500] + "..." if len(content) > 1500 else content

    # Format prediction result
    if label == "Fake":
        prediction_result = f"🚨 FAKE NEWS DETECTED"
    else:
        prediction_result = f"✅ NEWS APPEARS AUTHENTIC"

    confidence_text = f"{confidence:.1f}%"

    # Create detailed breakdown
    breakdown = {
        "Fake News Probability": f"{scores[0]:.2%}",
        "Real News Probability": f"{scores[1]:.2%}",
        "Final Classification": label,
        "Model Confidence": f"{confidence:.1f}%",
        "Content Length": f"{len(article_content)} characters",
        "Source URL": url
    }

    # Limit content preview
    content_preview = content[:2000] + "\n\n... (content truncated)" if len(content) > 2000 else content

    return prediction_result, confidence_text, breakdown, content_preview


# Launch the application
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,  # Default Gradio port
        share=True,  # Create shareable link
        debug=True,  # Enable debug mode
        show_error=True,  # Show detailed errors
        inbrowser=True  # Auto-open in browser
    )
