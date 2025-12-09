# Escape — Mood Engine & Realm Trigger API

## Overview

The Escape API provides mood inference from text using a fine-tuned RoBERTa transformer model, maps moods to immersive realms, and emits realm configuration packets for the Innerverse (Unreal Engine).

**Base URL:** `https://api.escapeapp.ai/v1`  
**Version:** 1.0.0

---

## Authentication

All endpoints (except `/health`) require API key authentication.

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `x-api-key` | string | Yes | Your API key |

**Example:**
```bash
curl -H "x-api-key: your-api-key" https://api.escapeapp.ai/v1/infer-mood
```

---

## Endpoints

### 1. Health Check

Check if the API is running.

```
GET /health
```

**Response:** `200 OK`
```json
{
  "status": "ok"
}
```

---

### 2. Infer Mood

Analyze text to infer the user's emotional state using the ML model.

```
POST /infer-mood
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | Unique user identifier |
| `text` | string | Yes | Text to analyze for mood |
| `context` | object | No | Optional contextual data |
| `source` | string | No | Source of input: `app`, `game`, `journal`, `lucille` |

**Context Object (optional):**
```json
{
  "time_of_day": "morning",
  "session_length": 1200,
  "device": "mobile",
  "recent_moods": ["Joy", "Sadness"]
}
```

**Example Request:**
```bash
curl -X POST https://api.escapeapp.ai/v1/infer-mood \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "user_id": "user_12345",
    "text": "I feel really anxious about my upcoming presentation",
    "source": "journal"
  }'
```

**Response:** `200 OK`
```json
{
  "mood": "Fear",
  "intensity": 0.89,
  "confidence": 0.94,
  "timestamp": "2025-12-09T15:30:00.000Z"
}
```

**Supported Moods:**

| Mood | Description |
|------|-------------|
| `Sadness` | Feelings of sorrow, grief, or melancholy |
| `Joy` | Happiness, excitement, or positive emotions |
| `Love` | Affection, warmth, or romantic feelings |
| `Anger` | Frustration, irritation, or rage |
| `Fear` | Anxiety, worry, or terror |
| `Surprise` | Astonishment or unexpected reactions |

---

### 3. Decide Realm

Map an inferred mood to an immersive realm and generate the Unreal Engine configuration packet.

```
POST /decide-realm
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | Unique user identifier |
| `mood` | string | Yes | Mood label from inference |
| `intensity` | float | Yes | Mood intensity (0.0 - 1.0) |
| `confidence` | float | No | Model confidence (0.0 - 1.0) |
| `context` | object | No | Optional contextual data |

**Example Request:**
```bash
curl -X POST https://api.escapeapp.ai/v1/decide-realm \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "user_id": "user_12345",
    "mood": "Fear",
    "intensity": 0.89,
    "confidence": 0.94
  }'
```

**Response:** `200 OK`
```json
{
  "user_id": "user_12345",
  "realm": "Shadowfall",
  "reason": "Fear (intensity 0.89, confidence 0.94) -> Shadowfall",
  "packet": {
    "realm": "Shadowfall",
    "weather": "RainMedium",
    "lighting": "FogLowBlue",
    "npc_profile": "SupportiveGuardian",
    "music": "Grounding_DarkPads",
    "session_id": "sess_a1b2c3d4"
  }
}
```

**Mood → Realm Mapping:**

| Mood | Realm | Weather | Lighting | NPC Profile | Music |
|------|-------|---------|----------|-------------|-------|
| Sadness | Misthollow | FogHeavy | DimBlue | CompassionateHealer | Melancholy_SoftStrings |
| Joy | Sunvale | ClearSunny | BrightWarm | PlayfulCompanion | Uplifting_BrightMelody |
| Love | Heartgarden | GentleBreeze | SoftPink | WarmEmpath | Tender_AcousticHarmony |
| Anger | Emberpeak | StormBrewing | HarshRed | CalmMediator | Intense_PowerDrums |
| Fear | Shadowfall | RainMedium | FogLowBlue | SupportiveGuardian | Grounding_DarkPads |
| Surprise | Wonderpeak | RainbowMist | DynamicShift | CuriousGuide | Mysterious_Chimes |

---

### 4. Emit Realm

Send the realm configuration packet to Unreal Engine or queue it for later delivery.

```
POST /emit-realm
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `target` | string | Yes | Delivery target: `unreal`, `pubsub`, `queue` |
| `packet` | object | Yes | The realm packet from `/decide-realm` |

**Target Options:**

| Target | Description | Response Status |
|--------|-------------|-----------------|
| `unreal` | Direct send to Unreal Engine | `sent` |
| `pubsub` | Publish to Pub/Sub topic | `published` |
| `queue` | Push to message queue | `queued` |

**Example Request:**
```bash
curl -X POST https://api.escapeapp.ai/v1/emit-realm \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "target": "unreal",
    "packet": {
      "realm": "Shadowfall",
      "weather": "RainMedium",
      "lighting": "FogLowBlue",
      "npc_profile": "SupportiveGuardian",
      "music": "Grounding_DarkPads",
      "session_id": "sess_a1b2c3d4"
    }
  }'
```

**Response:** `202 Accepted`
```json
{
  "status": "sent",
  "tx_id": "tx_5f296dab7e8c4a1fb2c3d4e5f6a7b8c9"
}
```

---

### 5. Debug Simulate

Test the complete pipeline without affecting production systems.

```
POST /debug/simulate
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `mode` | string | Yes | Simulation mode: `text`, `random`, `batch` |
| `payload` | object | No | Mode-specific payload |

**Modes:**

#### Text Mode
Simulate inference from a single text input.

```json
{
  "mode": "text",
  "payload": {
    "text": "I'm feeling really happy today!",
    "user_id": "test_user"
  }
}
```

**Response:**
```json
{
  "result": {
    "step_1_infer": {
      "mood": "Joy",
      "intensity": 0.98,
      "confidence": 0.99
    },
    "step_2_decide": {
      "realm": "Sunvale",
      "packet": {
        "realm": "Sunvale",
        "weather": "ClearSunny",
        "lighting": "BrightWarm",
        "npc_profile": "PlayfulCompanion",
        "music": "Uplifting_BrightMelody",
        "session_id": "sess_1234abcd"
      }
    },
    "step_3_emit": {
      "status": "simulated",
      "tx_id": "tx_sim_5678efgh"
    }
  }
}
```

#### Random Mode
Generate a random mood and realm for testing.

```json
{
  "mode": "random",
  "payload": {}
}
```

**Response:**
```json
{
  "result": {
    "random_mood": "Anger",
    "intensity": 0.72,
    "confidence": 0.85,
    "realm": "Emberpeak",
    "packet": {
      "realm": "Emberpeak",
      "weather": "StormBrewing",
      "lighting": "HarshRed",
      "npc_profile": "CalmMediator",
      "music": "Intense_PowerDrums",
      "session_id": "sess_abcd1234"
    }
  }
}
```

#### Batch Mode
Process multiple texts at once (max 10).

```json
{
  "mode": "batch",
  "payload": {
    "texts": [
      "I'm so excited for the weekend!",
      "I feel really down today.",
      "This makes me so angry!"
    ]
  }
}
```

**Response:**
```json
{
  "result": {
    "batch_results": [
      {"text": "I'm so excited for the weekend!", "mood": "Joy", "intensity": 0.95, "realm": "Sunvale"},
      {"text": "I feel really down today.", "mood": "Sadness", "intensity": 0.88, "realm": "Misthollow"},
      {"text": "This makes me so angry!", "mood": "Anger", "intensity": 0.91, "realm": "Emberpeak"}
    ]
  }
}
```

---

## Error Responses

### 401 Unauthorized
Missing or invalid API key.

```json
{
  "detail": "Invalid or missing API key"
}
```

### 422 Unprocessable Entity
Invalid request body or missing required fields.

```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error
Server-side error during processing.

```json
{
  "detail": "Mood inference failed: <error message>"
}
```

---

## Complete Pipeline Example

Here's how to use the full mood-to-realm pipeline:

```python
import requests

API_BASE = "https://api.escapeapp.ai/v1"
HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": "your-api-key"
}

# Step 1: Infer mood from user text
infer_response = requests.post(
    f"{API_BASE}/infer-mood",
    headers=HEADERS,
    json={
        "user_id": "user_12345",
        "text": "I'm feeling really anxious about tomorrow"
    }
)
mood_data = infer_response.json()
print(f"Mood: {mood_data['mood']} (confidence: {mood_data['confidence']})")

# Step 2: Decide realm based on mood
decide_response = requests.post(
    f"{API_BASE}/decide-realm",
    headers=HEADERS,
    json={
        "user_id": "user_12345",
        "mood": mood_data["mood"],
        "intensity": mood_data["intensity"],
        "confidence": mood_data["confidence"]
    }
)
realm_data = decide_response.json()
print(f"Realm: {realm_data['realm']}")

# Step 3: Emit realm packet to Unreal Engine
emit_response = requests.post(
    f"{API_BASE}/emit-realm",
    headers=HEADERS,
    json={
        "target": "unreal",
        "packet": realm_data["packet"]
    }
)
emit_data = emit_response.json()
print(f"Emitted: {emit_data['status']} (tx: {emit_data['tx_id']})")
```

---

## Model Information

The mood inference is powered by a fine-tuned **RoBERTa-base** transformer model.

| Property | Value |
|----------|-------|
| Base Model | `roberta-base` (125M parameters) |
| Training Dataset | `dair-ai/emotion` (16,000 samples) |
| Classes | 6 (Sadness, Joy, Love, Anger, Fear, Surprise) |
| Test Accuracy | 92.85% |
| F1 Score (weighted) | 0.928 |

**Per-Class Performance:**

| Mood | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| Sadness | 0.96 | 0.97 | 0.96 |
| Joy | 0.95 | 0.95 | 0.95 |
| Love | 0.85 | 0.81 | 0.83 |
| Anger | 0.94 | 0.91 | 0.93 |
| Fear | 0.85 | 0.95 | 0.89 |
| Surprise | 0.83 | 0.68 | 0.75 |

---

## Interactive Documentation

When running locally, access the interactive API docs at:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## Rate Limits

| Plan | Requests/Minute | Requests/Day |
|------|-----------------|--------------|
| Free | 10 | 100 |
| Developer | 60 | 5,000 |
| Production | 300 | 50,000 |

---

## Support

For API support, contact: **api-support@escapeapp.ai**

GitHub: https://github.com/epsilon403/mood_engine
