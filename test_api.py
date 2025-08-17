#!/usr/bin/env python3
"""
Test script for the simplified FastAPI JSON endpoints
"""
import requests
import json

# Base URL for your FastAPI application
BASE_URL = "http://localhost:5001"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def test_form_options():
    """Test getting form field options"""
    print("Testing Form Options...")
    response = requests.get(f"{BASE_URL}/options")
    print(f"Status: {response.status_code}")
    print(f"Available Options:")
    options = response.json()
    for field, values in options.items():
        print(f"  {field}: {values}")
    print("-" * 50)

def test_prediction_api():
    """Test the prediction API"""
    print("Testing Prediction API...")
    
    # Sample data for prediction
    prediction_data = {
        "gender": "female",
        "ethnicity": "group A",
        "parental_level_of_education": "bachelor's degree",
        "lunch": "standard",
        "test_preparation_course": "completed",
        "reading_score": 75.0,
        "writing_score": 66.0
    }
    
    print(f"Input Data: {json.dumps(prediction_data, indent=2)}")
    
    # Make prediction request
    response = requests.post(
        f"{BASE_URL}/predict",
        json=prediction_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction Result:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Status: {result['status']}")
        print(f"  Message: {result['message']}")
    else:
        print(f"Error: {response.text}")
    
    print("-" * 50)

def test_invalid_data():
    """Test API with invalid data"""
    print("Testing Invalid Data...")
    
    # Invalid data (missing required fields)
    invalid_data = {
        "gender": "female",
        "ethnicity": "group A"
        # Missing other required fields
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=invalid_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 422:
        print("Correctly rejected invalid data")
    else:
        print(f"Unexpected response: {response.text}")
    
    print("-" * 50)

if __name__ == "__main__":
    print("Academic Performance Analytics API - Testing Script")
    print("=" * 50)
    
    try:
        test_health_check()
        test_form_options()
        test_prediction_api()
        test_invalid_data()
        
        print("All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the FastAPI server.")
        print("Make sure your application is running on http://localhost:5001")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 