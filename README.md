# Escape — Mood Engine & Realm Trigger API

A production-ready API for real-time emotion detection and immersive realm generation, designed to power the **Innerverse** experience in Unreal Engine.

## 🎯 Overview

The Escape Mood Engine uses fine-tuned RoBERTa transformer models to:
1. **Infer emotions** from user text with 93% accuracy
2. **Map emotions to immersive realms** with custom environments
3. **Generate Unreal Engine packets** for real-time world adaptation

## 📁 Project Structure

```
mood_engine/
├── src/
│   └── api/
│       └── main.py          # FastAPI application
├── scripts/
│   ├── train_mood_model.py  # Model training script
│   └── run_tests.sh         # Test runner
├── tests/
│   ├── test_api.py          # Comprehensive tests
│   └── quick_test.py        # Quick demo tests
├── models/                   # Trained models (git-ignored)
│   └── mood_model/          # Fine-tuned RoBERTa
├── data/                     # Training outputs (git-ignored)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/epsilon403/mood_engine.git
cd mood_engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
python scripts/train_mood_model.py
```

This downloads RoBERTa-base and fine-tunes it on 16K emotion samples (~1 hour on GPU).

### Start the API

```bash
python src/api/main.py
```

API will be available at `http://localhost:8000`

### Test the API

```bash
# Run quick tests
python tests/quick_test.py

# Or use Swagger UI
open http://localhost:8000/docs
```

## 🌍 Realm Mapping

| Emotion | Realm | Environment |
|---------|-------|-------------|
| Sadness | Misthollow | Foggy, dim blue, healing NPCs |
| Joy | Sunvale | Sunny, bright warm, playful companions |
| Love | Heartgarden | Gentle breeze, soft pink, empathetic NPCs |
| Anger | Emberpeak | Stormy, harsh red, mediating NPCs |
| Fear | Shadowfall | Rainy, blue fog, supportive guardians |
| Surprise | Wonderpeak | Rainbow mist, dynamic lighting, curious guides |

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/infer-mood` | POST | Emotion inference from text |
| `/decide-realm` | POST | Map emotion to realm |
| `/emit-realm` | POST | Send packet to Unreal Engine |
| `/debug/simulate` | POST | Full pipeline simulation |

### Example Request

```bash
curl -X POST http://localhost:8000/infer-mood \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-key-change-in-production" \
  -d '{"user_id": "demo", "text": "I feel amazing today!"}'
```

### Response

```json
{
  "mood": "Joy",
  "intensity": 0.95,
  "confidence": 0.98,
  "timestamp": "2025-12-09T10:30:00Z"
}
```

## 📊 Model Performance

- **Architecture**: Fine-tuned RoBERTa-base
- **Dataset**: dair-ai/emotion (16,000 samples)
- **Test Accuracy**: 93.0%
- **Training Time**: ~1 hour (RTX 2000 Ada)

| Emotion | Accuracy | F1-Score |
|---------|----------|----------|
| Sadness | 96.9% | 96.6% |
| Joy | 95.4% | 95.4% |
| Love | 81.8% | 83.9% |
| Anger | 91.6% | 92.7% |
| Fear | 92.9% | 88.7% |
| Surprise | 66.7% | 72.7% |

## 🔐 Authentication

All endpoints require `x-api-key` header:

```bash
x-api-key: dev-key-change-in-production
```

Set custom key via environment variable:
```bash
export ESCAPE_API_KEY="your-secure-key"
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
