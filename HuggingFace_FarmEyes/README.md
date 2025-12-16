---
title: FarmEyes
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
suggested_hardware: cpu-basic
---

# 🌱 FarmEyes

**AI-Powered Crop Disease Detection for African Farmers**

[![Built for Awarri Challenge](https://img.shields.io/badge/Built%20for-Awarri%20Challenge%202025-green)](https://awarri.com)
[![N-ATLaS Powered](https://img.shields.io/badge/Powered%20by-N--ATLaS-blue)](https://huggingface.co/NCAIR1/N-ATLaS)

---

## 🎯 What is FarmEyes?

FarmEyes is an AI application that helps African farmers identify crop diseases and get treatment recommendations in their native languages. Simply upload a photo of your crop, and FarmEyes will:

1. **Detect** the disease using computer vision (YOLOv11)
2. **Diagnose** the condition with severity assessment
3. **Translate** all information to your preferred language
4. **Chat** with an AI assistant for follow-up questions

---

## 🌍 Supported Languages

| Language | Native Name |
|----------|-------------|
| 🇬🇧 English | English |
| 🇳🇬 Hausa | Yaren Hausa |
| 🇳🇬 Yoruba | Èdè Yorùbá |
| 🇳🇬 Igbo | Asụsụ Igbo |

---

## 🦠 Detectable Diseases

| Crop | Diseases |
|------|----------|
| 🌿 **Cassava** | Bacterial Blight, Mosaic Virus |
| 🍫 **Cocoa** | Monilia Disease, Phytophthora Disease |
| 🍅 **Tomato** | Gray Mold Disease, Wilt Disease |

---

## 🚀 How to Use

### Step 1: Select Language
Choose your preferred language from the welcome screen.

### Step 2: Upload Image
Take a photo of the affected crop leaf and upload it.

### Step 3: View Results
- Disease name and confidence score
- Severity level (Low/Moderate/High/Critical)
- Treatment recommendations
- Cost estimates in Nigerian Naira (₦)

### Step 4: Ask Questions
Use the chat feature to ask follow-up questions about the diagnosis.

---

## 🔧 Technology Stack

| Component | Technology |
|-----------|------------|
| **Disease Detection** | YOLOv11 (trained on African crops) |
| **Language Model** | N-ATLaS (Nigerian multilingual AI) |
| **Speech-to-Text** | OpenAI Whisper |
| **Backend** | FastAPI |
| **Frontend** | Custom HTML/CSS/JS |

---

## 📱 Features

- ✅ **Image Upload** - Drag & drop or click to upload
- ✅ **Real-time Detection** - Results in seconds
- ✅ **Multilingual Support** - 4 Nigerian languages
- ✅ **Voice Input** - Speak your questions
- ✅ **Text-to-Speech** - Listen to responses
- ✅ **Treatment Advice** - Practical farming guidance
- ✅ **Cost Estimates** - In Nigerian Naira

---

## ⚠️ First Startup Notice

**Please be patient on first use!**

The N-ATLaS language model (~4.92GB) is downloaded automatically on first startup. This may take **5-15 minutes** depending on connection speed. Subsequent uses will be much faster.

---

## 🏆 About

FarmEyes was built for the **Awarri Developer Challenge 2025** to address the critical need for accessible agricultural AI in Africa. 

**The Problem:**
- 20-80% crop losses annually due to diseases
- Only 1 extension worker per 10,000 farmers (FAO recommends 1:1,000)
- Agricultural knowledge locked in English

**Our Solution:**
- AI-powered disease detection accessible via smartphone
- Native language support through N-ATLaS
- Practical, localized treatment recommendations

---

## 👨‍💻 Developer

**Fola-AI**

- 🤗 HuggingFace: [@Fola-AI](https://huggingface.co/Fola-AI)

---

## 📄 License

Apache 2.0

---

## 🙏 Acknowledgments

- [NCAIR](https://ncair.nitda.gov.ng/) for N-ATLaS model
- [Ultralytics](https://ultralytics.com/) for YOLOv11
- [HuggingFace](https://huggingface.co/) for hosting
- [Awarri](https://awarri.com/) for the challenge opportunity

---

*Built with ❤️ for African Farmers*
