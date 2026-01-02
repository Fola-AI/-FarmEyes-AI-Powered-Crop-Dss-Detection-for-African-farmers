# 🌱 FarmEyes

### AI-Powered Multilingual Crop Health Assistant for African Farmers

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![N-ATLaS](https://img.shields.io/badge/N--ATLaS-NCAIR-orange.svg)](https://huggingface.co/NCAIR1/N-ATLaS)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-purple.svg)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Awarri Challenge](https://img.shields.io/badge/Awarri-Developer%20Challenge%202025-red.svg)](https://awarri.com)



---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Solution](#-solution)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Architecture](#-architecture)
- [N-ATLaS Integration](#-n-atlas-integration)
- [Screenshots](#-screenshots)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Disease Coverage](#-disease-coverage)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🌍 Overview

**FarmEyes** is an AI-powered crop health assistant that bridges the agricultural knowledge gap for African smallholder farmers. By combining computer vision disease detection with Nigeria's multilingual AI model (N-ATLaS), FarmEyes delivers actionable agricultural guidance in **Hausa**, **Yoruba**, **Igbo**, and **English**.

<!-- OVERVIEW DIAGRAM -->
<p align="center">
  <img src="assets/images/overview-diagram.png" alt="FarmEyes Overview" width="80%">
</p>

### Key Statistics

| Metric | Value |
|--------|-------|
| Target Users | 38+ Million African Smallholder Farmers |
| Languages Supported | 4 (English, Hausa, Yoruba, Igbo) |
| Diseases Detected | 6 Crop Diseases |
| Crops Covered | 3 (Cassava, Cocoa, Tomato) |
| N-ATLaS Contribution | 40-45% of Total Process |

---

## 🎯 Problem Statement

Africa's agricultural sector faces a critical yet invisible crisis: **a knowledge barrier that costs farmers billions in lost yields annually**.

<!-- PROBLEM INFOGRAPHIC -->
<p align="center">
  <img src="assets/images/problem-infographic.png" alt="Problem Statement" width="70%">
</p>

### The Challenge

- **70%** of rural African farmers cannot read or speak English fluently
- **20-40%** annual crop losses due to undiagnosed diseases
- **Limited access** to agricultural extension officers (1 officer per 3,000+ farmers)
- **Technical terminology** in existing solutions excludes most farmers
- **No voice support** for farmers with limited literacy

### The Gap

| Current State | Farmer Reality |
|---------------|----------------|
| Agricultural AI tools in English only | Farmers speak indigenous languages |
| Text-based interfaces | Many farmers have limited literacy |
| Generic global advice | Need locally relevant solutions |
| Costs in USD | Need prices in local currency (NGN) |
| Complex technical terms | Need simple, actionable guidance |

---

## 💡 Solution

FarmEyes combines **YOLOv11** computer vision with **N-ATLaS** multilingual AI to create an accessible agricultural assistant that speaks the farmer's language.

<!-- SOLUTION ARCHITECTURE -->
<p align="center">
  <img src="assets/images/solution-architecture.png" alt="Solution Architecture" width="85%">
</p>

### How It Works

```
📱 Farmer takes photo of diseased crop
    ↓
🔍 YOLOv11 detects and classifies disease
    ↓
🧠 N-ATLaS generates localized diagnosis report
    ↓
🗣️ Farmer receives advice in their native language
    ↓
🎤 Voice interaction for follow-up questions (Whisper + MMS-TTS)
```

### Value Proposition

| Without FarmEyes | With FarmEyes |
|------------------|---------------|
| English-only diagnosis | Native language support |
| Text-only interface | Voice input and output |
| Generic treatment advice | Nigerian-specific recommendations |
| Unknown costs | Prices in Nigerian Naira (₦) |
| Limited reach (~30% farmers) | Expanded reach (~70%+ farmers) |

---

## ✨ Features

### Core Features

<!-- FEATURE GRID -->
<p align="center">
  <img src="assets/images/features-grid.png" alt="Features Overview" width="90%">
</p>

#### 🔬 Disease Detection
- Real-time crop disease identification using YOLOv11s
- Confidence scoring with severity assessment
- Support for 6 diseases across 3 crops

#### 🌐 Multilingual Support
- Full interface localization in 4 languages
- N-ATLaS powered translation and generation
- Cultural adaptation of agricultural terminology

#### 🎤 Voice Interaction
- Speech-to-text input via OpenAI Whisper
- Text-to-speech output via Meta MMS-TTS
- Complete voice-based workflow for illiterate users

#### 💬 Contextual Chat
- AI-powered agricultural Q&A
- Diagnosis-aware responses
- Multi-turn conversation with memory

#### 📋 Treatment Recommendations
- Organic and chemical treatment options
- Cost estimates in Nigerian Naira (₦)
- Local market availability information

#### 🔄 Offline Capability
- Hybrid inference (API + local model)
- Works in low-connectivity rural areas
- Automatic fallback to local GGUF model

---

## 🛠️ Technology Stack

### AI Models

| Component | Technology | Purpose |
|-----------|------------|---------|
| Disease Detection | YOLOv11 | Crop disease classification |
| Language Model | N-ATLaS (NCAIR1/N-ATLaS) | Translation and text generation |
| Speech-to-Text | OpenAI Whisper | Voice input transcription |
| Text-to-Speech | Meta MMS-TTS | Multilingual audio output |

### Backend

| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | FastAPI | REST API and routing |
| Language | Python 3.10+ | Core application logic |
| Session Management | In-memory store | User context persistence |
| API Client | HuggingFace Router | N-ATLaS inference |

### Frontend

| Component | Technology | Purpose |
|-----------|------------|---------|
| Structure | HTML5 | Page layout |
| Styling | CSS3 (Dark Theme) | ChatGPT-inspired UI |
| Interactivity | Vanilla JavaScript | Client-side logic |
| Audio | Web Audio API | Voice recording |

### Deployment

| Component | Technology | Purpose |
|-----------|------------|---------|
| Current | HuggingFace Spaces | Demo hosting |
| Planned | Next.js | Production web app |
| Planned | React Native | iOS/Android mobile apps |

---

## 🏗️ Architecture

### System Architecture

<!-- SYSTEM ARCHITECTURE DIAGRAM -->
<p align="center">
  <img src="assets/images/system-architecture.png" alt="System Architecture" width="90%">
</p>

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Language   │  │  Diagnosis  │  │    Chat     │              │
│  │  Selector   │  │    Page     │  │    Page     │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└───────────────────────────┬─────────────────────────────────────┘
                            │ REST API
┌───────────────────────────┴─────────────────────────────────────┐
│                      FASTAPI BACKEND                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ /detect  │  │  /chat   │  │/transcribe│ │   /tts   │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
└───────┼─────────────┼─────────────┼─────────────┼───────────────┘
        │             │             │             │
┌───────┴─────────────┴─────────────┴─────────────┴───────────────┐
│                       SERVICE LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Disease    │  │     Chat     │  │   Whisper    │          │
│  │   Detector   │  │   Service    │  │   Service    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘          │
│         │                 │                                      │
│  ┌──────┴─────────────────┴──────────────────────────┐          │
│  │              N-ATLaS MODEL LAYER                   │          │
│  │  ┌─────────────────┐  ┌─────────────────┐         │          │
│  │  │ HuggingFace API │  │  Local GGUF     │         │          │
│  │  │    (Primary)    │  │   (Fallback)    │         │          │
│  │  └─────────────────┘  └─────────────────┘         │          │
│  └───────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

<!-- DATA FLOW DIAGRAM -->
<p align="center">
  <img src="assets/images/data-flow.png" alt="Data Flow" width="85%">
</p>

---

## 🧠 N-ATLaS Integration

**N-ATLaS** (Nigeria's Artificial Intelligence Language System) is the multilingual backbone of FarmEyes, handling approximately **40-45%** of the total application process.

### What is N-ATLaS?

| Specification | Details |
|---------------|---------|
| Base Architecture | Llama-3 8B (fine-tuned) |
| Developer | NCAIR (Nigerian Centre for AI Research) |
| Languages | English, Hausa, Yoruba, Igbo |
| Context Length | 8,092 tokens |
| Model Repository | [NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS) |

### N-ATLaS Functions in FarmEyes

<!-- NATLAS FUNCTIONS DIAGRAM -->
<p align="center">
  <img src="assets/images/natlas-functions.png" alt="N-ATLaS Functions" width="80%">
</p>

| Function | Description | Coverage |
|----------|-------------|----------|
| Disease Name Translation | Converts disease names to local languages | 100% |
| Symptom Translation | Translates symptom descriptions | 100% |
| Treatment Generation | Creates localized treatment advice | 100% |
| Prevention Advice | Translates prevention recommendations | 100% |
| Chat Responses | Generates contextual Q&A responses | 100% |
| UI Localization | Dynamically translates interface elements | 100% |

### Hybrid Inference Strategy

FarmEyes implements a robust hybrid approach for N-ATLaS inference:

```python
# Primary: HuggingFace Router API (Serverless)
API_URL = "https://router.huggingface.co/v1/chat/completions"

# Fallback: Local GGUF Model (Q4_K_M quantization)
LOCAL_MODEL = "N-ATLaS-GGUF-Q4_K_M.gguf"  # ~4.9GB
```

This ensures the application works even when:
- Internet connectivity is limited
- HuggingFace API is at capacity
- Users are in rural areas with poor network coverage

---

## 📸 Screenshots

### Language Selection

<!-- LANGUAGE SELECTION SCREENSHOT -->
<p align="center">
  <img src="assets/screenshots/language-selection.png" alt="Language Selection" width="70%">
</p>

*Users select their preferred language: English, Hausa, Yoruba, or Igbo*

### Disease Diagnosis

<!-- DIAGNOSIS SCREENSHOT -->
<p align="center">
  <img src="assets/screenshots/diagnosis-page.png" alt="Diagnosis Page" width="70%">
</p>

*Upload a crop image to receive instant disease detection with confidence scoring*

### Diagnosis Results

<!-- RESULTS SCREENSHOT -->
<p align="center">
  <img src="assets/screenshots/diagnosis-results.png" alt="Diagnosis Results" width="70%">
</p>

*Detailed results showing disease name, severity, symptoms, and treatment options*

### Treatment Information

<!-- TREATMENT SCREENSHOT -->
<p align="center">
  <img src="assets/screenshots/treatment-tab.png" alt="Treatment Tab" width="70%">
</p>

*Treatment recommendations with costs in Nigerian Naira (₦)*

### Chat Interface

<!-- CHAT SCREENSHOT -->
<p align="center">
  <img src="assets/screenshots/chat-interface.png" alt="Chat Interface" width="70%">
</p>

*AI-powered chat for follow-up questions in your native language*

### Voice Input

<!-- VOICE INPUT SCREENSHOT -->
<p align="center">
  <img src="assets/screenshots/voice-input.png" alt="Voice Input" width="70%">
</p>

*Voice recording for farmers who prefer speaking over typing*

### Mobile View

<!-- MOBILE SCREENSHOT -->
<p align="center">
  <img src="assets/screenshots/mobile-view.png" alt="Mobile View" width="40%">
</p>

*Responsive design optimized for mobile devices*

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git
- HuggingFace account and API token
- 8GB+ RAM (16GB recommended for local model)
- For Apple Silicon: MPS acceleration supported

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/FarmEyes.git
cd FarmEyes

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your HuggingFace token

# 5. Run the application
python main.py
```

### Environment Variables

Create a `.env` file in the project root:

```env
# HuggingFace API Token (Required)
HF_TOKEN=your_huggingface_token_here

# Server Configuration
HOST=0.0.0.0
PORT=7860
DEBUG=false

# Model Configuration
USE_LOCAL_MODEL=false
LOCAL_MODEL_PATH=./models/N-ATLaS-GGUF-Q4_K_M.gguf
```

### Get HuggingFace Token

1. Create account at [huggingface.co](https://huggingface.co/join)
2. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Create new token with `read` permissions
4. Accept N-ATLaS terms at [NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS)

### Installation for Apple Silicon (M1/M2/M3)

```bash
# Install with Metal Performance Shaders (MPS) support
pip install torch torchvision torchaudio

# Install llama-cpp-python with Metal support
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# Verify MPS is available
python -c "import torch; print(torch.backends.mps.is_available())"
```

---

## 📖 Usage

### Starting the Application

```bash
# Development mode
python main.py

# Production mode with Uvicorn
uvicorn main:app --host 0.0.0.0 --port 7860 --workers 4
```

The application will be available at `http://localhost:7860`

### User Flow

<!-- USER FLOW DIAGRAM -->
<p align="center">
  <img src="assets/images/user-flow.png" alt="User Flow" width="80%">
</p>

1. **Select Language**: Choose English, Hausa, Yoruba, or Igbo
2. **Upload Image**: Take or upload a photo of your crop
3. **View Results**: See disease diagnosis with details and treatment
4. **Ask Questions**: Chat with AI assistant for more information
5. **Listen**: Use voice output to hear responses (optional)

### Voice Interaction

```
🎤 Voice Input Flow:
User speaks → Whisper (STT) → Text → N-ATLaS → Response

🔊 Voice Output Flow:
Response → N-ATLaS → Text → MMS-TTS → Audio playback
```

---

## 📚 API Documentation

### Base URL

```
http://localhost:7860/api
```

### Endpoints

#### Disease Detection

```http
POST /api/detect/
Content-Type: multipart/form-data

Parameters:
- file: Image file (JPEG, PNG)
- language: Language code (en, ha, yo, ig)
- session_id: Session identifier

Response:
{
  "success": true,
  "disease": "Cassava Mosaic Virus",
  "confidence": 0.93,
  "severity": "high",
  "report": { ... }
}
```

#### Chat

```http
POST /api/chat/
Content-Type: application/json

{
  "message": "How do I treat this disease?",
  "session_id": "abc123",
  "language": "ha"
}

Response:
{
  "success": true,
  "response": "Don magance cutar...",
  "session_id": "abc123"
}
```

#### Speech-to-Text

```http
POST /api/transcribe/
Content-Type: multipart/form-data

Parameters:
- file: Audio file (WAV, WebM)
- session_id: Session identifier

Response:
{
  "success": true,
  "text": "Transcribed text here",
  "language": "en"
}
```

#### Text-to-Speech

```http
POST /api/tts/
Content-Type: application/json

{
  "text": "Text to convert to speech",
  "language": "ha"
}

Response:
{
  "success": true,
  "audio_base64": "UklGRi...",
  "content_type": "audio/flac"
}
```

### Interactive API Docs

Visit `http://localhost:7860/api/docs` for Swagger UI documentation.

<!-- API DOCS SCREENSHOT -->
<p align="center">
  <img src="assets/screenshots/api-docs.png" alt="API Documentation" width="80%">
</p>

---

## 📁 Project Structure

```
FarmEyes/
├── main.py                      # Application entry point
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── README.md                    # This file
│
├── api/                         # API Layer
│   ├── __init__.py
│   └── routes/
│       ├── __init__.py
│       ├── detection.py         # /api/detect endpoint
│       ├── chat.py              # /api/chat endpoint
│       ├── transcribe.py        # /api/transcribe endpoint
│       └── tts.py               # /api/tts endpoint
│
├── models/                      # AI Models
│   ├── __init__.py
│   ├── natlas_model.py          # N-ATLaS integration (Hybrid API + GGUF)
│   ├── yolo_model.py            # YOLOv11 disease detection
│   └── farmeyes_yolov11.pt      # Trained YOLO weights
│
├── services/                    # Business Logic
│   ├── __init__.py
│   ├── disease_detector.py      # Detection service
│   ├── diagnosis_generator.py   # Report generation
│   ├── translator.py            # Translation service
│   ├── chat_service.py          # Contextual chat
│   ├── session_manager.py       # Session handling
│   ├── whisper_service.py       # STT service
│   └── tts_service.py           # TTS service
│
├── utils/                       # Utilities
│   ├── __init__.py
│   └── prompt_templates.py      # N-ATLaS prompts
│
├── data/                        # Static Data
│   └── knowledge_base.json      # Disease information
│
├── static/                      # Static Assets
│   └── ui_translations.json     # UI text translations
│
├── frontend/                    # Frontend Application
│   ├── index.html               # SPA entry point
│   ├── css/
│   │   ├── main.css             # Core styles
│   │   ├── theme-dark.css       # Dark theme
│   │   └── pages.css            # Page-specific styles
│   └── js/
│       ├── api.js               # API client
│       ├── i18n.js              # Internationalization
│       ├── voice.js             # Voice recording
│       ├── tts.js               # Text-to-speech
│       ├── diagnosis.js         # Diagnosis page
│       ├── chat.js              # Chat interface
│       └── app.js               # Main controller
│
├── assets/                      # Documentation Assets
│   ├── images/                  # Diagrams and graphics
│   └── screenshots/             # Application screenshots
│
└── docs/                        # Additional Documentation
    ├── TECHNICAL_ANALYSIS.pdf   # N-ATLaS technical analysis
    ├── API_REFERENCE.md         # Detailed API docs
    └── DEPLOYMENT.md            # Deployment guide
```

---

## 🦠 Disease Coverage

FarmEyes currently detects **6 diseases** across **3 crops**:

### Cassava Diseases

<!-- CASSAVA DISEASES IMAGE -->
<p align="center">
  <img src="assets/images/cassava-diseases.png" alt="Cassava Diseases" width="80%">
</p>

| Disease | Severity | Yield Impact | Key Symptoms |
|---------|----------|--------------|--------------|
| Cassava Bacterial Blight | Moderate-Severe | 20-100% | Angular leaf spots, wilting, gum exudation |
| Cassava Mosaic Virus | Moderate-Severe | 32-69% | Yellow/green mosaic patterns, leaf distortion |

### Cocoa Diseases

<!-- COCOA DISEASES IMAGE -->
<p align="center">
  <img src="assets/images/cocoa-diseases.png" alt="Cocoa Diseases" width="80%">
</p>

| Disease | Severity | Yield Impact | Key Symptoms |
|---------|----------|--------------|--------------|
| Monilia Disease (Frosty Pod Rot) | Severe | 40-80% | White fungal growth, pod rot |
| Phytophthora Disease (Black Pod) | Severe | 30-90% | Dark lesions, pod blackening |

### Tomato Diseases

<!-- TOMATO DISEASES IMAGE -->
<p align="center">
  <img src="assets/images/tomato-diseases.png" alt="Tomato Diseases" width="80%">
</p>

| Disease | Severity | Yield Impact | Key Symptoms |
|---------|----------|--------------|--------------|
| Gray Mold (Botrytis) | Moderate-Severe | 20-50% | Gray fuzzy growth, stem lesions |
| Wilt Disease | Severe | 50-100% | Yellowing, wilting, vascular browning |

---

## 🗺️ Roadmap

### Current Version (v1.0)

- [x] YOLOv11 disease detection (6 diseases)
- [x] N-ATLaS multilingual support (4 languages)
- [x] FastAPI backend with REST API
- [x] Custom HTML/CSS/JS frontend
- [x] Voice input (Whisper STT)
- [x] Voice output (MMS-TTS)
- [x] Contextual chat assistant
- [x] HuggingFace Spaces deployment

### Phase 1: Production Ready (Q1 2026)

- [ ] Next.js migration for improved performance
- [ ] PostgreSQL database integration
- [ ] User authentication system
- [ ] Streaming N-ATLaS responses
- [ ] Response caching optimization
- [ ] CI/CD pipeline setup

### Phase 2: Mobile Apps (Q2 2026)

- [ ] React Native mobile application
- [ ] iOS App Store deployment
- [ ] Google Play Store deployment
- [ ] Offline disease detection
- [ ] Push notifications for disease alerts
- [ ] Camera integration optimization

### Phase 3: Expansion (Q3-Q4 2026)

- [ ] Additional languages (Swahili, Amharic)
- [ ] More crops (Maize, Rice, Yam, Pepper)
- [ ] 15+ new diseases
- [ ] Community features
- [ ] Expert consultation booking
- [ ] Market price integration

<!-- ROADMAP TIMELINE -->
<p align="center">
  <img src="assets/images/roadmap-timeline.png" alt="Roadmap Timeline" width="90%">
</p>

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

- 🐛 **Bug Reports**: Found a bug? Open an issue
- 💡 **Feature Requests**: Have an idea? Share it with us
- 📝 **Documentation**: Help improve our docs
- 🌍 **Translations**: Help translate to more languages
- 💻 **Code**: Submit pull requests

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/FarmEyes.git
cd FarmEyes

# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes and commit
git add .
git commit -m "Add: description of your changes"

# Push and create a pull request
git push origin feature/your-feature-name
```

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add comments for complex logic
- Write tests for new features

### Pull Request Guidelines

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Keep commits focused and atomic
5. Write clear commit messages

---

## 🙏 Acknowledgments

### Organizations

- **[NCAIR](https://ncair.nitda.gov.ng/)** - For developing the N-ATLaS model
- **[Awarri](https://awarri.com/)** - For hosting the Developer Challenge 2025
- **[HuggingFace](https://huggingface.co/)** - For model hosting and inference API
- **[Ultralytics](https://ultralytics.com/)** - For YOLOv11

### Open Source Projects

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [Meta MMS](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) - Multilingual speech
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Local LLM inference

### Datasets

- [PlantVillage Dataset](https://plantvillage.psu.edu/) - Plant disease images
- [Cassava Disease Dataset](https://www.kaggle.com/c/cassava-disease) - Cassava-specific images

### Special Thanks

- Nigerian farming communities for inspiration and feedback
- Agricultural extension officers for domain expertise
- Beta testers across Nigeria

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 FarmEyes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📞 Contact

- **Project Lead**: [Afolabi AJao]
- **Email**: [afolinks@outlook.com]
- **LinkedIn**: [www.linkedin.com/in/afolabi-ajao-ab8442a7)

---

<p align="center">
  <b>Built with ❤️ for African Farmers</b>
</p>

<p align="center">
  <img src="assets/images/footer-logo.png" alt="FarmEyes Logo" width="150">
</p>

<p align="center">
  <a href="#-farmeyes">Back to Top ↑</a>
</p>
