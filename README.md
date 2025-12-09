# Escape — Mood Engine & Realm Trigger API

A production-ready API for real-time emotion detection and immersive realm generation, designed to power the **Innerverse** experience in Unreal Engine.

## 🎯 Overview

The Escape Mood Engine uses fine-tuned RoBERTa transformer models to:
1. **Infer emotions** from user text with 93% accuracy
2. **Map emotions to immersive realms** with custom environments
3. **Generate Unreal Engine packets** for real-time world adaptation

---

## 📁 Project Structure

```
mood_engine/
├── src/
│   └── api/
│       └── main.py              # FastAPI application
├── scripts/
│   ├── train_mood_model.py      # Model training script
│   └── run_tests.sh             # Automated test runner
├── tests/
│   ├── test_api.py              # Comprehensive API tests
│   └── quick_test.py            # Quick demo tests
├── models/                       # Trained models (git-ignored)
├── data/                         # Training outputs (git-ignored)
├── docs/                         # Additional documentation
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## 📄 File Documentation

### `src/api/main.py`
**FastAPI Application** - The main API server for mood inference and realm generation.

| Component | Description |
|-----------|-------------|
| **Enums** | `SourceEnum`, `TargetEnum`, `SimulationMode` for request validation |
| **Pydantic Models** | Request/Response schemas for all endpoints |
| **REALM_MAPPING** | Dictionary mapping emotions → realm configurations |
| **MoodModelManager** | Singleton class that loads and manages the RoBERTa model |
| **Endpoints** | `/health`, `/infer-mood`, `/decide-realm`, `/emit-realm`, `/debug/simulate` |

**Key Functions:**
- `verify_api_key()` - API key authentication middleware
- `infer_mood()` - Text → Emotion classification using RoBERTa
- `decide_realm()` - Emotion → Realm mapping with UE packet generation
- `emit_realm()` - Send packets to Unreal Engine/PubSub/Queue

---

### `scripts/train_mood_model.py`
**Model Training Script** - Fine-tunes RoBERTa-base on emotion classification.

| Component | Description |
|-----------|-------------|
| **Configuration** | `MODEL_NAME`, `OUTPUT_DIR`, `NUM_EPOCHS`, `BATCH_SIZE` |
| **compute_metrics()** | Calculates accuracy, precision, recall, F1 during training |
| **train_model()** | Main training pipeline with evaluation and visualization |

**Training Pipeline:**
1. Load `dair-ai/emotion` dataset (16,000 samples)
2. Tokenize text using RoBERTa tokenizer
3. Fine-tune all model layers for 3 epochs
4. Evaluate on test set with comprehensive metrics
5. Generate confusion matrix and save model

**Outputs:**
- `models/mood_model/` - Trained model weights and tokenizer
- `data/confusion_matrix.png` - Visual performance matrix
- `data/training_metrics.txt` - Detailed accuracy report

---

### `scripts/run_tests.sh`
**Test Runner Script** - Bash script to start API and run tests.

```bash
#!/bin/bash
# Starts API server in background
# Runs comprehensive test suite
# Cleans up processes on completion
```

---

### `tests/test_api.py`
**Comprehensive Test Suite** - Full API testing framework.

| Test Category | Tests |
|---------------|-------|
| **Health** | Server connectivity, health endpoint |
| **Mood Inference** | All 6 emotions with expected predictions |
| **Realm Decision** | Mood → Realm mapping validation |
| **Emit Realm** | Packet emission to all targets |
| **Debug Simulate** | Text, random, and batch simulation modes |
| **Full Pipeline** | End-to-end: infer → decide → emit |
| **Error Handling** | Missing API key, invalid key, malformed requests |

**Class:** `EscapeAPITester`
- `run_all_tests()` - Execute complete test suite with summary

---

### `tests/quick_test.py`
**Quick Demo Script** - Interactive API demonstration.

| Function | Description |
|----------|-------------|
| `test_mood_inference()` | Test 6 different emotion texts |
| `test_full_pipeline()` | Complete infer→decide→emit flow |
| `test_simulation()` | Debug simulation endpoint demo |

---

### `requirements.txt`
**Python Dependencies**

| Category | Packages |
|----------|----------|
| **Core ML** | torch, transformers, datasets, accelerate |
| **API** | fastapi, uvicorn, pydantic |
| **Metrics** | scikit-learn, numpy |
| **Visualization** | matplotlib, seaborn |
| **Testing** | requests, pytest |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/epsilon403/mood_engine.git
cd mood_engine

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Train the Model

```bash
python scripts/train_mood_model.py
```

**Output:** Fine-tuned model saved to `models/mood_model/`

### Start the API

```bash
python src/api/main.py
```

**Server:** `http://localhost:8000`  
**Docs:** `http://localhost:8000/docs`

### Run Tests

```bash
# Quick demo
python tests/quick_test.py

# Comprehensive tests
python tests/test_api.py

# Automated (starts server + runs tests)
./scripts/run_tests.sh
```

---

## 🌍 Realm Mapping

| Emotion | Realm | Weather | Lighting | NPC Profile | Music |
|---------|-------|---------|----------|-------------|-------|
| **Sadness** | Misthollow | FogHeavy | DimBlue | CompassionateHealer | Melancholy_SoftStrings |
| **Joy** | Sunvale | ClearSunny | BrightWarm | PlayfulCompanion | Uplifting_BrightMelody |
| **Love** | Heartgarden | GentleBreeze | SoftPink | WarmEmpath | Tender_AcousticHarmony |
| **Anger** | Emberpeak | StormBrewing | HarshRed | CalmMediator | Intense_PowerDrums |
| **Fear** | Shadowfall | RainMedium | FogLowBlue | SupportiveGuardian | Grounding_DarkPads |
| **Surprise** | Wonderpeak | RainbowMist | DynamicShift | CuriousGuide | Mysterious_Chimes |

---

## 📡 API Reference

### Authentication
All endpoints (except `/health`) require:
```
x-api-key: dev-key-change-in-production
```

### Endpoints

#### `GET /health`
Health check - no authentication required.

**Response:**
```json
{"status": "ok"}
```

---

#### `POST /infer-mood`
Infer emotion from text using ML model.

**Request:**
```json
{
  "user_id": "string",
  "text": "string",
  "source": "app|game|journal|lucille",
  "context": {}
}
```

**Response:**
```json
{
  "mood": "Joy",
  "intensity": 0.95,
  "confidence": 0.98,
  "timestamp": "2025-12-09T10:30:00Z"
}
```

---

#### `POST /decide-realm`
Map emotion to realm configuration.

**Request:**
```json
{
  "user_id": "string",
  "mood": "Joy",
  "intensity": 0.95,
  "confidence": 0.98
}
```

**Response:**
```json
{
  "user_id": "string",
  "realm": "Sunvale",
  "reason": "Joy (intensity 0.95, confidence 0.98) -> Sunvale",
  "packet": {
    "realm": "Sunvale",
    "weather": "ClearSunny",
    "lighting": "BrightWarm",
    "npc_profile": "PlayfulCompanion",
    "music": "Uplifting_BrightMelody",
    "session_id": "sess_abc123"
  }
}
```

---

#### `POST /emit-realm`
Send realm packet to target system.

**Request:**
```json
{
  "target": "unreal|pubsub|queue",
  "packet": {}
}
```

**Response (202):**
```json
{
  "status": "sent|published|queued",
  "tx_id": "tx_abc123..."
}
```

---

#### `POST /debug/simulate`
Simulate full pipeline for testing.

**Modes:**
- `text` - Infer mood from provided text
- `random` - Generate random mood/realm
- `batch` - Process multiple texts

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Architecture** | RoBERTa-base (125M params) |
| **Dataset** | dair-ai/emotion |
| **Training Samples** | 16,000 |
| **Test Accuracy** | 93.0% |
| **Weighted F1** | 92.9% |
| **Training Time** | ~1 hour (RTX 2000 Ada) |

### Per-Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Sadness | 96.2% | 96.9% | 96.6% | 581 |
| Joy | 95.4% | 95.4% | 95.4% | 695 |
| Love | 86.1% | 81.8% | 83.9% | 159 |
| Anger | 93.7% | 91.6% | 92.7% | 275 |
| Fear | 84.9% | 92.9% | 88.7% | 224 |
| Surprise | 80.0% | 66.7% | 72.7% | 66 |

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ESCAPE_API_KEY` | `dev-key-change-in-production` | API authentication key |
| `MODEL_PATH` | `./models/mood_model` | Path to trained model |

### Training Configuration

Edit `scripts/train_mood_model.py`:
```python
MODEL_NAME = "roberta-base"    # Base model
OUTPUT_DIR = "./models/mood_model"  # Save location
NUM_EPOCHS = 3                 # Training epochs
BATCH_SIZE = 16                # Batch size
```

---

## 🎮 Unreal Engine Integration

The API generates JSON packets ready for UE consumption:

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

**Integration options:**
- Direct HTTP calls from UE
- WebSocket connection (coming soon)
- Message queue (Redis/RabbitMQ)
- Google Pub/Sub

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with ❤️ for the Innerverse experience**
