#!/usr/bin/env python3
"""
Quick Test Demo for Escape Mood Engine API
"""

import requests
import json

# API Configuration
API_URL = "http://localhost:8000"
API_KEY = "dev-key-change-in-production"
headers = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY
}

def test_mood_inference():
    """Test mood inference with different emotions"""
    
    test_cases = [
        "I am absolutely thrilled about this new opportunity!",
        "I feel so sad and empty inside today.",
        "I'm really angry about how they treated me.",
        "I'm terrified about the upcoming surgery.",
        "I love spending time with my family so much.",
        "Wow, I never expected that plot twist!"
    ]
    
    print("🧠 MOOD INFERENCE TESTS")
    print("=" * 50)
    
    for i, text in enumerate(test_cases, 1):
        try:
            payload = {
                "user_id": "demo_user",
                "text": text,
                "source": "app"
            }
            
            response = requests.post(f"{API_URL}/infer-mood", headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n{i}. Text: \"{text}\"")
                print(f"   🎭 Mood: {result['mood']}")
                print(f"   📊 Intensity: {result['intensity']:.2f}")
                print(f"   🎯 Confidence: {result['confidence']:.2f}")
            else:
                print(f"\n{i}. ❌ Error: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"\n{i}. ❌ Error: {str(e)}")

def test_full_pipeline():
    """Test the complete pipeline"""
    
    print("\n\n🔄 FULL PIPELINE TEST")
    print("=" * 50)
    
    test_text = "I feel incredibly nervous about my job interview tomorrow"
    
    try:
        # Step 1: Infer mood
        print(f"\n📝 Input Text: \"{test_text}\"")
        
        mood_payload = {
            "user_id": "pipeline_user",
            "text": test_text,
            "source": "app"
        }
        
        mood_response = requests.post(f"{API_URL}/infer-mood", headers=headers, json=mood_payload)
        
        if mood_response.status_code == 200:
            mood_result = mood_response.json()
            print(f"\n🧠 Step 1 - Mood Inference:")
            print(f"   Mood: {mood_result['mood']}")
            print(f"   Intensity: {mood_result['intensity']}")
            print(f"   Confidence: {mood_result['confidence']}")
            
            # Step 2: Decide realm
            realm_payload = {
                "user_id": "pipeline_user",
                "mood": mood_result["mood"],
                "intensity": mood_result["intensity"],
                "confidence": mood_result["confidence"]
            }
            
            realm_response = requests.post(f"{API_URL}/decide-realm", headers=headers, json=realm_payload)
            
            if realm_response.status_code == 200:
                realm_result = realm_response.json()
                print(f"\n🌍 Step 2 - Realm Decision:")
                print(f"   Realm: {realm_result['realm']}")
                print(f"   Reason: {realm_result['reason']}")
                print(f"   Packet: {json.dumps(realm_result['packet'], indent=6)}")
                
                # Step 3: Emit realm
                emit_payload = {
                    "target": "unreal",
                    "packet": realm_result["packet"]
                }
                
                emit_response = requests.post(f"{API_URL}/emit-realm", headers=headers, json=emit_payload)
                
                if emit_response.status_code == 202:
                    emit_result = emit_response.json()
                    print(f"\n🚀 Step 3 - Realm Emission:")
                    print(f"   Status: {emit_result['status']}")
                    print(f"   Transaction ID: {emit_result['tx_id']}")
                    print(f"\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
                else:
                    print(f"\n❌ Step 3 failed: HTTP {emit_response.status_code}")
            else:
                print(f"\n❌ Step 2 failed: HTTP {realm_response.status_code}")
        else:
            print(f"\n❌ Step 1 failed: HTTP {mood_response.status_code}")
            
    except Exception as e:
        print(f"\n❌ Pipeline error: {str(e)}")

def test_simulation():
    """Test debug simulation endpoint"""
    
    print("\n\n🎮 SIMULATION TEST")
    print("=" * 50)
    
    payload = {
        "mode": "text",
        "payload": {
            "text": "I'm absolutely delighted with the results!",
            "user_id": "simulation_test"
        }
    }
    
    try:
        response = requests.post(f"{API_URL}/debug/simulate", headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()["result"]
            print(f"\n🎯 Simulation Result:")
            print(f"   Step 1 (Infer): {result['step_1_infer']}")
            print(f"   Step 2 (Decide): Realm = {result['step_2_decide']['realm']}")
            print(f"   Step 3 (Emit): Status = {result['step_3_emit']['status']}")
            print(f"\n✅ SIMULATION COMPLETED!")
        else:
            print(f"\n❌ Simulation failed: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"\n❌ Simulation error: {str(e)}")

if __name__ == "__main__":
    print("🚀 ESCAPE API DEMO")
    print("🔗 Testing API at:", API_URL)
    print("🔑 Using API Key:", API_KEY[:10] + "...")
    
    # Check if API is running
    try:
        health_response = requests.get(f"{API_URL}/health")
        if health_response.status_code == 200:
            print("✅ API is running!\n")
        else:
            print("❌ API health check failed")
            exit(1)
    except:
        print("❌ Cannot connect to API. Make sure it's running with: python main.py")
        exit(1)
    
    # Run tests
    test_mood_inference()
    test_full_pipeline()
    test_simulation()
    
    print("\n\n🎉 ALL TESTS COMPLETED!")
    print("📚 Try the interactive docs at: http://localhost:8000/docs")