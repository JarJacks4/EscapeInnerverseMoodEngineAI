"""
Escape — Mood Engine & Realm Trigger API
FastAPI implementation based on OpenAPI 3.0.3 specification
"""

import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = "./saved_mood_model"
API_KEY = os.getenv("ESCAPE_API_KEY", "dev-key-change-in-production")

# ============================================================================
# Enums
# ============================================================================

class SourceEnum(str, Enum):
    app = "app"
    game = "game"
    journal = "journal"
    lucille = "lucille"

class TargetEnum(str, Enum):
    unreal = "unreal"
    pubsub = "pubsub"
    queue = "queue"

class SimulationMode(str, Enum):
    text = "text"
    random = "random"
    batch = "batch"

# ============================================================================
# Pydantic Models (Request/Response)
# ============================================================================

# Health
class HealthResponse(BaseModel):
    status: str = "ok"

# Infer Mood
class InferMoodRequest(BaseModel):
    user_id: str
    text: str
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional contextual fields (time_of_day, session_length, device, recent_moods)")
    source: Optional[SourceEnum] = None

class InferMoodResponse(BaseModel):
    mood: str
    intensity: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    timestamp: datetime

# Decide Realm
class DecideRealmRequest(BaseModel):
    user_id: str
    mood: str
    intensity: float
    confidence: Optional[float] = None
    context: Optional[Dict[str, Any]] = None

class RealmPacket(BaseModel):
    realm: str
    weather: str
    lighting: str
    npc_profile: str
    music: str
    session_id: str

class DecideRealmResponse(BaseModel):
    user_id: str
    realm: str
    reason: str
    packet: RealmPacket

# Emit Realm
class EmitRealmRequest(BaseModel):
    target: TargetEnum
    packet: Dict[str, Any]

class EmitRealmResponse(BaseModel):
    status: str
    tx_id: str

# Debug Simulate
class SimulateRequest(BaseModel):
    mode: SimulationMode
    payload: Optional[Dict[str, Any]] = None

class SimulateResponse(BaseModel):
    result: Dict[str, Any]

# ============================================================================
# Mood -> Realm Mapping Configuration
# ============================================================================

REALM_MAPPING = {
    "Sadness": {
        "realm": "Misthollow",
        "weather": "FogHeavy",
        "lighting": "DimBlue",
        "npc_profile": "CompassionateHealer",
        "music": "Melancholy_SoftStrings"
    },
    "Joy": {
        "realm": "Sunvale",
        "weather": "ClearSunny",
        "lighting": "BrightWarm",
        "npc_profile": "PlayfulCompanion",
        "music": "Uplifting_BrightMelody"
    },
    "Love": {
        "realm": "Heartgarden",
        "weather": "GentleBreeze",
        "lighting": "SoftPink",
        "npc_profile": "WarmEmpath",
        "music": "Tender_AcousticHarmony"
    },
    "Anger": {
        "realm": "Emberpeak",
        "weather": "StormBrewing",
        "lighting": "HarshRed",
        "npc_profile": "CalmMediator",
        "music": "Intense_PowerDrums"
    },
    "Fear": {
        "realm": "Shadowfall",
        "weather": "RainMedium",
        "lighting": "FogLowBlue",
        "npc_profile": "SupportiveGuardian",
        "music": "Grounding_DarkPads"
    },
    "Surprise": {
        "realm": "Wonderpeak",
        "weather": "RainbowMist",
        "lighting": "DynamicShift",
        "npc_profile": "CuriousGuide",
        "music": "Mysterious_Chimes"
    }
}

# Fallback for moods not in the trained model
DEFAULT_REALM = {
    "realm": "Neutralis",
    "weather": "PartlyCloudy",
    "lighting": "NaturalDaylight",
    "npc_profile": "NeutralObserver",
    "music": "Ambient_Nature"
}

# ============================================================================
# ML Model Manager (Singleton)
# ============================================================================

class MoodModelManager:
    _instance = None
    _model = None
    _tokenizer = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self):
        """Load the fine-tuned RoBERTa model"""
        if self._model is not None:
            return

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading mood model on {self._device}...")

        try:
            self._tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
            self._model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
            self._model.to(self._device)
            self._model.eval()
            print("Mood model loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load fine-tuned model from {MODEL_PATH}: {e}")
            print("Loading base roberta-base model instead...")
            self._tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self._model = RobertaForSequenceClassification.from_pretrained(
                "roberta-base",
                num_labels=6,
                id2label={0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"},
                label2id={"Sadness": 0, "Joy": 1, "Love": 2, "Anger": 3, "Fear": 4, "Surprise": 5}
            )
            self._model.to(self._device)
            self._model.eval()

    def infer_mood(self, text: str) -> tuple[str, float, float]:
        """
        Infer mood from text.
        Returns: (mood_label, intensity, confidence)
        """
        if self._model is None:
            self.load_model()

        # Tokenize input
        inputs = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self._device)

        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        # Get prediction
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

        # Get mood label from model config
        mood_label = self._model.config.id2label[predicted_class]

        # Intensity based on max probability (how strongly the mood is expressed)
        intensity = confidence  # Can be refined with more sophisticated logic

        return mood_label, intensity, confidence


# Global model manager
model_manager = MoodModelManager()

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Escape — Mood Engine & Realm Trigger API",
    version="1.0.0",
    description="API for receiving user inputs, inferring mood, mapping mood -> realm, and emitting realm trigger packets for the Innerverse (Unreal Engine)."
)

# API Key Security
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load ML model on startup"""
    model_manager.load_model()

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="ok")


@app.post("/infer-mood", response_model=InferMoodResponse, tags=["Mood Engine"])
async def infer_mood(request: InferMoodRequest, api_key: str = Depends(verify_api_key)):
    """
    Infer mood from text/metadata (calls ML model).
    
    Accepts text and lightweight behavioral signals. Returns mood label, intensity, confidence.
    """
    try:
        mood, intensity, confidence = model_manager.infer_mood(request.text)

        return InferMoodResponse(
            mood=mood,
            intensity=round(intensity, 2),
            confidence=round(confidence, 2),
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mood inference failed: {str(e)}")


@app.post("/decide-realm", response_model=DecideRealmResponse, tags=["Realm Engine"])
async def decide_realm(request: DecideRealmRequest, api_key: str = Depends(verify_api_key)):
    """
    Map mood to realm and create realm packet.
    
    Accepts a mood inference (or raw input) and returns the selected realm and UE packet.
    """
    # Get realm configuration based on mood
    realm_config = REALM_MAPPING.get(request.mood, DEFAULT_REALM)

    # Generate session ID
    session_id = f"sess_{uuid.uuid4().hex[:8]}"

    # Build packet
    packet = RealmPacket(
        realm=realm_config["realm"],
        weather=realm_config["weather"],
        lighting=realm_config["lighting"],
        npc_profile=realm_config["npc_profile"],
        music=realm_config["music"],
        session_id=session_id
    )

    # Build reason string
    confidence_str = f", confidence {request.confidence}" if request.confidence else ""
    reason = f"{request.mood} (intensity {request.intensity}{confidence_str}) -> {realm_config['realm']}"

    return DecideRealmResponse(
        user_id=request.user_id,
        realm=realm_config["realm"],
        reason=reason,
        packet=packet
    )


@app.post("/emit-realm", response_model=EmitRealmResponse, status_code=202, tags=["Realm Engine"])
async def emit_realm(request: EmitRealmRequest, api_key: str = Depends(verify_api_key)):
    """
    Emit realm packet to UE (or queue it).
    
    Sends the packet to the Unreal Engine endpoint (or pushes to Pub/Sub / webhook).
    """
    tx_id = f"tx_{uuid.uuid4().hex}"

    # TODO: Implement actual emission logic based on target
    # - unreal: Send to Unreal Engine WebSocket/HTTP endpoint
    # - pubsub: Publish to Google Pub/Sub or similar
    # - queue: Push to message queue (RabbitMQ, Redis, etc.)

    if request.target == TargetEnum.unreal:
        # Placeholder for Unreal Engine integration
        status = "sent"
    elif request.target == TargetEnum.pubsub:
        # Placeholder for Pub/Sub integration
        status = "published"
    else:
        # Queue
        status = "queued"

    return EmitRealmResponse(status=status, tx_id=tx_id)


@app.post("/debug/simulate", response_model=SimulateResponse, tags=["Debug"])
async def debug_simulate(request: SimulateRequest, api_key: str = Depends(verify_api_key)):
    """
    Simulate flow (test only).
    
    Accepts test payloads for simulating infer->decide->emit pipeline.
    """
    result = {}

    if request.mode == SimulationMode.text:
        # Simulate text-based mood inference
        text = request.payload.get("text", "I feel great today!") if request.payload else "I feel great today!"
        user_id = request.payload.get("user_id", "test_user") if request.payload else "test_user"

        # Infer mood
        mood, intensity, confidence = model_manager.infer_mood(text)

        # Decide realm
        realm_config = REALM_MAPPING.get(mood, DEFAULT_REALM)
        session_id = f"sess_{uuid.uuid4().hex[:8]}"

        result = {
            "step_1_infer": {
                "mood": mood,
                "intensity": round(intensity, 2),
                "confidence": round(confidence, 2)
            },
            "step_2_decide": {
                "realm": realm_config["realm"],
                "packet": {
                    **realm_config,
                    "session_id": session_id
                }
            },
            "step_3_emit": {
                "status": "simulated",
                "tx_id": f"tx_sim_{uuid.uuid4().hex[:8]}"
            }
        }

    elif request.mode == SimulationMode.random:
        # Random mood simulation
        import random
        moods = list(REALM_MAPPING.keys())
        mood = random.choice(moods)
        intensity = round(random.uniform(0.3, 1.0), 2)
        confidence = round(random.uniform(0.6, 1.0), 2)

        realm_config = REALM_MAPPING[mood]
        session_id = f"sess_{uuid.uuid4().hex[:8]}"

        result = {
            "random_mood": mood,
            "intensity": intensity,
            "confidence": confidence,
            "realm": realm_config["realm"],
            "packet": {
                **realm_config,
                "session_id": session_id
            }
        }

    elif request.mode == SimulationMode.batch:
        # Batch simulation
        texts = request.payload.get("texts", []) if request.payload else []
        batch_results = []

        for text in texts[:10]:  # Limit to 10 for safety
            mood, intensity, confidence = model_manager.infer_mood(text)
            realm_config = REALM_MAPPING.get(mood, DEFAULT_REALM)
            batch_results.append({
                "text": text[:50] + "..." if len(text) > 50 else text,
                "mood": mood,
                "intensity": round(intensity, 2),
                "realm": realm_config["realm"]
            })

        result = {"batch_results": batch_results}

    return SimulateResponse(result=result)


# ============================================================================
# Run with Uvicorn
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
