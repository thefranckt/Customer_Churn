#!/usr/bin/env python3
"""
Test script to verify API structure without requiring trained models
"""

import sys
import json
from pathlib import Path

def test_api_structure():
    """Test API file structure without running it"""
    
    print("🧪 Testing API Structure...")
    
    api_file = Path("api/deployment_api.py")
    
    if not api_file.exists():
        print("❌ API file not found")
        return False
    
    try:
        with open(api_file, 'r', encoding='utf-8') as f:
            api_code = f.read()
        
        # Check essential components
        checks = [
            ("FastAPI import", "from fastapi import FastAPI"),
            ("Pydantic import", "from pydantic import BaseModel"),
            ("Joblib import", "import joblib"),
            ("API instance", "api = FastAPI"),
            ("Customer schema", "class CustomerFeatures"),
            ("Health endpoint", "@api.get(\"/\")"),
            ("Prediction endpoint", "@api.post(\"/predict\")"),
            ("Model loading", "joblib.load"),
        ]
        
        for name, pattern in checks:
            if pattern in api_code:
                print(f"✅ {name}: Found")
            else:
                print(f"❌ {name}: Missing")
                return False
        
        # Check expected input fields
        expected_fields = [
            "tenure", "MonthlyCharges", "TotalCharges",
            "Contract_Two_year", "InternetService_Fiber_optic",
            "OnlineSecurity_No", "TechSupport_No", "PaperlessBilling"
        ]
        
        missing_fields = []
        for field in expected_fields:
            if field not in api_code:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"⚠️  Missing expected fields: {missing_fields}")
        else:
            print("✅ All expected input fields found")
        
        print("✅ API structure validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error reading API file: {e}")
        return False

def test_notebooks_structure():
    """Test if notebooks have the expected structure"""
    
    print("\n📚 Testing Notebooks Structure...")
    
    notebooks_dir = Path("notebooks")
    if not notebooks_dir.exists():
        print("❌ Notebooks directory not found")
        return False
    
    expected_notebooks = [
        "01_data_loading.ipynb",
        "02_eda.ipynb", 
        "03_preprocessing.ipynb",
        "04_modeling.ipynb",
        "05_evaluation.ipynb",
        "06_deployment.ipynb"
    ]
    
    all_exist = True
    for notebook in expected_notebooks:
        notebook_path = notebooks_dir / notebook
        if notebook_path.exists():
            print(f"✅ {notebook}: Found")
        else:
            print(f"❌ {notebook}: Missing")
            all_exist = False
    
    return all_exist

def create_sample_request():
    """Create a sample API request for testing"""
    
    sample_data = {
        "tenure": 12.0,
        "MonthlyCharges": 65.5,
        "TotalCharges": 786.0,
        "Contract_Two_year": 0,
        "InternetService_Fiber_optic": 1,
        "OnlineSecurity_No": 1,
        "TechSupport_No": 1,
        "PaperlessBilling": 1
    }
    
    print("\n📋 Sample API Request:")
    print("POST /predict")
    print("Content-Type: application/json")
    print()
    print(json.dumps(sample_data, indent=2))
    print()
    print("💡 Use this data to test your API once models are trained!")
    
    return sample_data

def main():
    print("🔧 CUSTOMER CHURN API STRUCTURE TEST")
    print("=" * 50)
    
    api_ok = test_api_structure()
    notebooks_ok = test_notebooks_structure()
    
    if api_ok and notebooks_ok:
        print("\n🎉 All structure tests passed!")
        create_sample_request()
        
        print("\n🚀 Ready to proceed with:")
        print("   1. Run notebooks 01-04 to train models")
        print("   2. Start API: uvicorn api.deployment_api:api --reload")
        print("   3. Test API at: http://localhost:8000/docs")
        
        return True
    else:
        print("\n❌ Some structure tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)