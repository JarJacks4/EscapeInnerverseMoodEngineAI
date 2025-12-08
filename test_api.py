#!/usr/bin/env python3
"""
Test Suite for Escape Mood Engine API
Comprehensive testing script with multiple test scenarios
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "dev-key-change-in-production"
HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY
}

class EscapeAPITester:
    def __init__(self):
        self.base_url = API_BASE_URL
        self.headers = HEADERS
        self.test_results = []

    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results"""
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"   Details: {details}")
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details
        })
        print()

    def test_health_check(self):
        """Test the health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health")
            success = response.status_code == 200 and response.json().get("status") == "ok"
            self.log_test("Health Check", success, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Health Check", False, f"Error: {str(e)}")

    def test_mood_inference_basic(self):
        """Test basic mood inference"""
        test_cases = [
            ("I am so happy today!", "Joy"),
            ("I feel really sad and lonely", "Sadness"),
            ("I'm furious about this situation", "Anger"),
            ("I'm scared of what might happen", "Fear"),
            ("I love spending time with my family", "Love"),
            ("Wow, I didn't expect that at all!", "Surprise")
        ]

        for text, expected_mood in test_cases:
            try:
                payload = {
                    "user_id": "test_user",
                    "text": text,
                    "source": "app"
                }
                response = requests.post(
                    f"{self.base_url}/infer-mood",
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    predicted_mood = result["mood"]
                    confidence = result["confidence"]
                    intensity = result["intensity"]
                    
                    success = predicted_mood == expected_mood
                    details = f"Text: '{text[:30]}...', Expected: {expected_mood}, Got: {predicted_mood}, Confidence: {confidence:.2f}"
                    self.log_test(f"Mood Inference - {expected_mood}", success, details)
                else:
                    self.log_test(f"Mood Inference - {expected_mood}", False, f"HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_test(f"Mood Inference - {expected_mood}", False, f"Error: {str(e)}")

    def test_realm_decision(self):
        """Test realm decision endpoint"""
        test_cases = [
            ("Sadness", 0.85, "Misthollow"),
            ("Joy", 0.92, "Sunvale"),
            ("Anger", 0.78, "Emberpeak"),
            ("Fear", 0.81, "Shadowfall"),
            ("Love", 0.88, "Heartgarden"),
            ("Surprise", 0.75, "Wonderpeak")
        ]

        for mood, intensity, expected_realm in test_cases:
            try:
                payload = {
                    "user_id": "test_user",
                    "mood": mood,
                    "intensity": intensity,
                    "confidence": 0.90
                }
                response = requests.post(
                    f"{self.base_url}/decide-realm",
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    selected_realm = result["realm"]
                    success = selected_realm == expected_realm
                    details = f"Mood: {mood}, Expected: {expected_realm}, Got: {selected_realm}"
                    self.log_test(f"Realm Decision - {mood}", success, details)
                else:
                    self.log_test(f"Realm Decision - {mood}", False, f"HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_test(f"Realm Decision - {mood}", False, f"Error: {str(e)}")

    def test_emit_realm(self):
        """Test realm emission endpoint"""
        targets = ["unreal", "pubsub", "queue"]
        
        for target in targets:
            try:
                payload = {
                    "target": target,
                    "packet": {
                        "realm": "Sunvale",
                        "weather": "ClearSunny",
                        "lighting": "BrightWarm",
                        "npc_profile": "PlayfulCompanion",
                        "music": "Uplifting_BrightMelody",
                        "session_id": "test_session_123"
                    }
                }
                response = requests.post(
                    f"{self.base_url}/emit-realm",
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code == 202:
                    result = response.json()
                    success = "tx_id" in result and "status" in result
                    details = f"Target: {target}, Status: {result.get('status')}, TX ID: {result.get('tx_id')[:10]}..."
                    self.log_test(f"Emit Realm - {target}", success, details)
                else:
                    self.log_test(f"Emit Realm - {target}", False, f"HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_test(f"Emit Realm - {target}", False, f"Error: {str(e)}")

    def test_debug_simulate(self):
        """Test debug simulation endpoint"""
        
        # Test text simulation
        try:
            payload = {
                "mode": "text",
                "payload": {
                    "text": "I feel absolutely amazing and full of energy!",
                    "user_id": "sim_user"
                }
            }
            response = requests.post(
                f"{self.base_url}/debug/simulate",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                success = all(step in result["result"] for step in ["step_1_infer", "step_2_decide", "step_3_emit"])
                details = f"Mode: text, Mood: {result['result']['step_1_infer']['mood']}"
                self.log_test("Debug Simulate - Text Mode", success, details)
            else:
                self.log_test("Debug Simulate - Text Mode", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test("Debug Simulate - Text Mode", False, f"Error: {str(e)}")

        # Test random simulation
        try:
            payload = {"mode": "random"}
            response = requests.post(
                f"{self.base_url}/debug/simulate",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                success = "random_mood" in result["result"] and "realm" in result["result"]
                details = f"Mode: random, Mood: {result['result']['random_mood']}"
                self.log_test("Debug Simulate - Random Mode", success, details)
            else:
                self.log_test("Debug Simulate - Random Mode", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test("Debug Simulate - Random Mode", False, f"Error: {str(e)}")

        # Test batch simulation
        try:
            payload = {
                "mode": "batch",
                "payload": {
                    "texts": [
                        "I'm so excited about this new opportunity!",
                        "This makes me feel really upset and frustrated.",
                        "I'm worried something bad will happen."
                    ]
                }
            }
            response = requests.post(
                f"{self.base_url}/debug/simulate",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                success = "batch_results" in result["result"] and len(result["result"]["batch_results"]) == 3
                details = f"Mode: batch, Processed: {len(result['result']['batch_results'])} texts"
                self.log_test("Debug Simulate - Batch Mode", success, details)
            else:
                self.log_test("Debug Simulate - Batch Mode", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test("Debug Simulate - Batch Mode", False, f"Error: {str(e)}")

    def test_full_pipeline(self):
        """Test the complete pipeline: infer -> decide -> emit"""
        try:
            test_text = "I feel absolutely terrified about the upcoming presentation"
            
            # Step 1: Infer mood
            infer_payload = {
                "user_id": "pipeline_test_user",
                "text": test_text,
                "source": "app"
            }
            infer_response = requests.post(
                f"{self.base_url}/infer-mood",
                headers=self.headers,
                json=infer_payload
            )
            
            if infer_response.status_code != 200:
                self.log_test("Full Pipeline", False, f"Inference failed: {infer_response.status_code}")
                return
                
            mood_result = infer_response.json()
            
            # Step 2: Decide realm
            decide_payload = {
                "user_id": "pipeline_test_user",
                "mood": mood_result["mood"],
                "intensity": mood_result["intensity"],
                "confidence": mood_result["confidence"]
            }
            decide_response = requests.post(
                f"{self.base_url}/decide-realm",
                headers=self.headers,
                json=decide_payload
            )
            
            if decide_response.status_code != 200:
                self.log_test("Full Pipeline", False, f"Realm decision failed: {decide_response.status_code}")
                return
                
            realm_result = decide_response.json()
            
            # Step 3: Emit realm
            emit_payload = {
                "target": "unreal",
                "packet": realm_result["packet"].__dict__ if hasattr(realm_result["packet"], '__dict__') else realm_result["packet"]
            }
            emit_response = requests.post(
                f"{self.base_url}/emit-realm",
                headers=self.headers,
                json=emit_payload
            )
            
            success = emit_response.status_code == 202
            details = f"Text: '{test_text[:30]}...', Mood: {mood_result['mood']}, Realm: {realm_result['realm']}"
            self.log_test("Full Pipeline", success, details)
            
        except Exception as e:
            self.log_test("Full Pipeline", False, f"Error: {str(e)}")

    def test_error_cases(self):
        """Test error handling"""
        
        # Test missing API key
        try:
            response = requests.post(
                f"{self.base_url}/infer-mood",
                headers={"Content-Type": "application/json"},
                json={"user_id": "test", "text": "test"}
            )
            success = response.status_code == 401
            self.log_test("Error Handling - Missing API Key", success, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Error Handling - Missing API Key", False, f"Error: {str(e)}")
        
        # Test invalid API key
        try:
            invalid_headers = self.headers.copy()
            invalid_headers["x-api-key"] = "invalid-key"
            response = requests.post(
                f"{self.base_url}/infer-mood",
                headers=invalid_headers,
                json={"user_id": "test", "text": "test"}
            )
            success = response.status_code == 401
            self.log_test("Error Handling - Invalid API Key", success, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Error Handling - Invalid API Key", False, f"Error: {str(e)}")
        
        # Test malformed request
        try:
            response = requests.post(
                f"{self.base_url}/infer-mood",
                headers=self.headers,
                json={"user_id": "test"}  # Missing required 'text' field
            )
            success = response.status_code == 422
            self.log_test("Error Handling - Malformed Request", success, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Error Handling - Malformed Request", False, f"Error: {str(e)}")

    def run_all_tests(self):
        """Run the complete test suite"""
        print("🧪 ESCAPE API TEST SUITE")
        print("=" * 50)
        print(f"Testing API at: {self.base_url}")
        print(f"Using API Key: {API_KEY}")
        print("=" * 50)
        print()
        
        # Wait for API to be ready
        print("⏳ Waiting for API to be ready...")
        for _ in range(10):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    print("✅ API is ready!\n")
                    break
            except:
                time.sleep(1)
        else:
            print("❌ API not responding. Make sure the server is running.\n")
            return
        
        # Run test suites
        print("🔍 RUNNING TESTS...")
        print("-" * 30)
        
        self.test_health_check()
        self.test_mood_inference_basic()
        self.test_realm_decision()
        self.test_emit_realm()
        self.test_debug_simulate()
        self.test_full_pipeline()
        self.test_error_cases()
        
        # Summary
        print("📊 TEST SUMMARY")
        print("=" * 50)
        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("🎉 All tests passed! Your API is working perfectly!")
        else:
            print("⚠️  Some tests failed. Check the details above.")
            failed_tests = [result for result in self.test_results if not result["success"]]
            print("\nFailed Tests:")
            for test in failed_tests:
                print(f"  - {test['test']}: {test['details']}")


if __name__ == "__main__":
    tester = EscapeAPITester()
    tester.run_all_tests()