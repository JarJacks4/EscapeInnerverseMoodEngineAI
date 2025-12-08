# Escape — Mood Engine & Realm Trigger API

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![RoBERTa](https://img.shields.io/badge/Model-RoBERTa--base-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A production-ready API for real-time emotion detection and immersive realm generation, designed to power the **Innerverse** experience in Unreal Engine.

## 🎯 Overview

The Escape Mood Engine uses fine-tuned RoBERTa transformer models to:
1. **Infer emotions** from user text with 93% accuracy
2. **Map emotions to immersive realms** with custom environments
3. **Generate Unreal Engine packets** for real-time world adaptation

## 🚀 Features

### 🧠 **AI-Powered Mood Detection**
- **Fine-tuned RoBERTa model** trained on 16K emotion samples
- **6 emotion categories**: Sadness, Joy, Love, Anger, Fear, Surprise
- **93% test accuracy** with confidence scoring
- **GPU-accelerated inference** (CUDA support)

### 🌍 **Dynamic Realm Mapping**
| Emotion | Realm | Environment |
|---------|-------|-------------|
| **Sadness** | Misthollow | Foggy, dim blue lighting, healing NPCs |
| **Joy** | Sunvale | Sunny, bright warm lighting, playful companions |
| **Love** | Heartgarden | Gentle breeze, soft pink lighting, empathetic NPCs |
| **Anger** | Emberpeak | Stormy, harsh red lighting, mediating NPCs |
| **Fear** | Shadowfall | Rainy, blue fog lighting, supportive guardians |
| **Surprise** | Wonderpeak | Rainbow mist, dynamic lighting, curious guides |

### 🔌 **Production-Ready API**
- **RESTful FastAPI** with OpenAPI 3.0.3 specification
- **API key authentication** for secure access
- **Comprehensive error handling** and validation
- **Real-time WebSocket support** (ready for UE integration)
- **Swagger UI documentation** at `/docs`

## 📦 Installation

### Prerequisites
- Python 3.12+
- NVIDIA GPU (recommended for training)
- 8GB+ RAM

### Setup
```bash
# Clone the repository
git clone https://github.com/epsilon403/mood_engine.git
cd mood_engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (optional - pre-trained model included)
python train_mood_model.py

# Start the API server
python main.py
```

## 🧪 Testing

### Quick Test
```bash
# Start API server
python main.py

# Run comprehensive tests
python quick_test.py

# Or use the automated test suite
./run_tests.sh
```

### Interactive Testing
Visit `http://localhost:8000/docs` for Swagger UI documentation and interactive API testing.

### Sample API Call
```bash
curl -X POST http://localhost:8000/infer-mood \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-key-change-in-production" \
  -d '{
    "user_id": "demo_user",
    "text": "I feel absolutely amazing today!",
    "source": "app"
  }'
```

**Response:**
```json
{
  "mood": "Joy",
  "intensity": 0.95,
  "confidence": 0.98,
  "timestamp": "2025-12-09T10:30:00.000Z"
}
```

## 📊 Model Performance

### Training Results
- **Dataset**: dair-ai/emotion (16,000 samples)
- **Architecture**: Fine-tuned RoBERTa-base
- **Training Time**: ~1 hour on RTX 2000 Ada
- **Final Accuracy**: 93.0%

### Per-Class Performance
| Emotion | Precision | Recall | F1-Score | Accuracy |
|---------|-----------|--------|----------|----------|
| Sadness | 96.24% | 96.90% | 96.57% | 96.90% |
| Joy | 95.40% | 95.40% | 95.40% | 95.40% |
| Love | 86.09% | 81.76% | 83.87% | 81.76% |
| Anger | 93.68% | 91.64% | 92.65% | 91.64% |
| Fear | 84.90% | 92.86% | 88.70% | 92.86% |
| Surprise | 80.00% | 66.67% | 72.73% | 66.67% |

## 🔧 API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `POST /infer-mood` - Emotion inference from text
- `POST /decide-realm` - Map emotion to realm configuration
- `POST /emit-realm` - Send realm packet to Unreal Engine
- `POST /debug/simulate` - Full pipeline simulation

### Authentication
All endpoints require the `x-api-key` header:
```bash
x-api-key: your-api-key-here
```

## 🎮 Unreal Engine Integration

The API generates structured packets for seamless UE integration:

```json
{
  "realm": "Shadowfall",
  "weather": "RainMedium",
  "lighting": "FogLowBlue", 
  "npc_profile": "SupportiveGuardian",
  "music": "Grounding_DarkPads",
  "session_id": "sess_abc123ef"
}
```

## 🗂️ Project Structure

```
mood_engine/
├── main.py                 # FastAPI application
├── train_mood_model.py     # Model training script
├── test_api.py            # Comprehensive test suite
├── quick_test.py          # Quick demo tests
├── run_tests.sh           # Automated test runner
├── requirements.txt       # Python dependencies
├── saved_mood_model/      # Trained model artifacts
├── results/               # Training checkpoints
├── confusion_matrix.png   # Model performance visualization
└── training_metrics.txt   # Training results
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the transformers library and RoBERTa model
- **dair-ai/emotion** dataset for training data
- **FastAPI** for the excellent web framework
- **Unreal Engine** for the target integration platform

---

**Built with ❤️ for the Innerverse experience**