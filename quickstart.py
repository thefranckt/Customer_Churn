#!/usr/bin/env python3
"""
Quick start script for Customer Churn project
"""

import os
import subprocess
import sys

def check_requirements():
    """Check if required files exist"""
    required_files = [
        "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "notebooks/04_modeling.ipynb"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found!")
    return True

def run_notebooks():
    """Run the notebooks in sequence"""
    notebooks = [
        "01_data_loading.ipynb",
        "02_eda.ipynb", 
        "03_preprocessing.ipynb",
        "04_modeling.ipynb"
    ]
    
    print("📚 Running notebooks in sequence...")
    for notebook in notebooks:
        notebook_path = f"notebooks/{notebook}"
        print(f"🔄 Running {notebook}...")
        
        try:
            result = subprocess.run([
                "jupyter", "nbconvert", 
                "--to", "notebook", 
                "--execute", 
                "--inplace", 
                notebook_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {notebook} completed successfully")
            else:
                print(f"❌ {notebook} failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("❌ Jupyter not found. Please install: pip install jupyter")
            return False
    
    return True

def start_api():
    """Start the FastAPI server"""
    print("🚀 Starting API server...")
    try:
        subprocess.run(["uvicorn", "api.deployment_api:api", "--reload"])
    except FileNotFoundError:
        print("❌ uvicorn not found. Please install: pip install uvicorn")
        return False
    except KeyboardInterrupt:
        print("\n👋 API server stopped")

def main():
    print("🏁 Customer Churn Project - Quick Start")
    print("=" * 40)
    
    if not check_requirements():
        print("\n💡 Please ensure all required files are in place before running.")
        return
    
    choice = input("\n🤔 What would you like to do?\n"
                  "1. Run all notebooks and start API\n"
                  "2. Just start API (models must exist)\n"
                  "3. Just run notebooks\n"
                  "Choice (1-3): ")
    
    if choice == "1":
        if run_notebooks():
            start_api()
    elif choice == "2":
        if os.path.exists("models/logreg_model.pkl"):
            start_api()
        else:
            print("❌ No trained model found. Please run notebooks first.")
    elif choice == "3":
        run_notebooks()
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()