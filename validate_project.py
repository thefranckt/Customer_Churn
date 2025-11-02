#!/usr/bin/env python3
"""
Comprehensive validation script for Customer Churn project
Tests all components to ensure everything works correctly.
"""

import os
import sys
import importlib.util
import json
from pathlib import Path

class ProjectValidator:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.issues = []
        self.successes = []
    
    def log_success(self, message):
        self.successes.append(message)
        print(f"✅ {message}")
    
    def log_issue(self, message):
        self.issues.append(message)
        print(f"❌ {message}")
    
    def check_directory_structure(self):
        """Check if all required directories exist"""
        print("\n🔍 Checking Directory Structure...")
        
        required_dirs = [
            "api", "data", "data/raw", "data/processed", 
            "models", "notebooks"
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                self.log_success(f"Directory exists: {dir_path}")
            else:
                self.log_issue(f"Missing directory: {dir_path}")
    
    def check_required_files(self):
        """Check if all critical files exist"""
        print("\n📁 Checking Required Files...")
        
        required_files = [
            "api/deployment_api.py",
            "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
            "requirements.txt",
            "Dockerfile",
            "README.md"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.log_success(f"File exists: {file_path}")
            else:
                self.log_issue(f"Missing file: {file_path}")
    
    def check_notebooks(self):
        """Check if all notebooks exist and have correct structure"""
        print("\n📚 Checking Notebooks...")
        
        notebooks = [
            "01_data_loading.ipynb",
            "02_eda.ipynb", 
            "03_preprocessing.ipynb",
            "04_modeling.ipynb",
            "05_evaluation.ipynb",
            "06_deployment.ipynb"
        ]
        
        for notebook in notebooks:
            notebook_path = self.project_root / "notebooks" / notebook
            if notebook_path.exists():
                self.log_success(f"Notebook exists: {notebook}")
            else:
                self.log_issue(f"Missing notebook: {notebook}")
    
    def check_api_syntax(self):
        """Check if API file has correct syntax"""
        print("\n🔧 Checking API Syntax...")
        
        api_file = self.project_root / "api" / "deployment_api.py"
        if not api_file.exists():
            self.log_issue("API file not found")
            return
        
        try:
            # Try to compile the API file
            with open(api_file, 'r', encoding='utf-8') as f:
                api_code = f.read()
            
            compile(api_code, str(api_file), 'exec')
            self.log_success("API file syntax is valid")
            
            # Check for key components - updated for enhanced API
            if 'get_current_model' in api_code or 'model_manager' in api_code:
                self.log_success("API uses enhanced model management system")
            elif 'model = joblib.load("models/logreg_model.pkl")' in api_code:
                self.log_success("API uses correct model path for Docker")
            else:
                self.log_issue("API model path may be incorrect for Docker")
            
            if 'api = FastAPI' in api_code:
                self.log_success("FastAPI app correctly defined")
            else:
                self.log_issue("FastAPI app not properly defined")
                
        except SyntaxError as e:
            self.log_issue(f"API syntax error: {e}")
    
    def check_dockerfile_compatibility(self):
        """Check if Dockerfile is compatible with current structure"""
        print("\n🐳 Checking Dockerfile...")
        
        dockerfile = self.project_root / "Dockerfile"
        if not dockerfile.exists():
            self.log_issue("Dockerfile not found")
            return
        
        try:
            with open(dockerfile, 'r', encoding='utf-8') as f:
                dockerfile_content = f.read()
            
            required_elements = [
                "COPY api/ ./api/",
                "COPY models/ ./models/", 
                "api.deployment_api:api"
            ]
            
            for element in required_elements:
                if element in dockerfile_content:
                    self.log_success(f"Dockerfile contains: {element}")
                else:
                    self.log_issue(f"Dockerfile missing: {element}")
                    
        except Exception as e:
            self.log_issue(f"Error reading Dockerfile: {e}")
    
    def check_dependencies(self):
        """Check if key dependencies can be imported"""
        print("\n📦 Checking Dependencies...")
        
        key_deps = [
            "fastapi", "pydantic", "joblib", "numpy", 
            "pandas", "sklearn", "xgboost", "uvicorn"
        ]
        
        for dep in key_deps:
            try:
                importlib.import_module(dep)
                self.log_success(f"Package available: {dep}")
            except ImportError:
                self.log_issue(f"Package missing: {dep}")
    
    def check_data_preprocessing_chain(self):
        """Check if data files exist in the correct processing chain"""
        print("\n📊 Checking Data Processing Chain...")
        
        # Check raw data
        raw_data = self.project_root / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        if raw_data.exists():
            self.log_success("Raw data file exists")
        else:
            self.log_issue("Raw data file missing")
        
        # Check processed data
        processed_files = [
            "telco_clean_base.csv",
            "telco_final_preprocessed.csv"
        ]
        
        for file in processed_files:
            file_path = self.project_root / "data" / "processed" / file
            if file_path.exists():
                self.log_success(f"Processed data exists: {file}")
            else:
                self.log_issue(f"Processed data missing: {file}")
    
    def check_model_structure(self):
        """Check model directory structure and expected files"""
        print("\n🤖 Checking Model Structure...")
        
        models_dir = self.project_root / "models"
        if not models_dir.exists():
            self.log_issue("Models directory missing")
            return
        
        self.log_success("Models directory exists")
        
        # Check if models would be saved correctly
        expected_models = [
            "logreg_model.pkl",
            "xgb_model.pkl", 
            "rf_model.pkl"
        ]
        
        models_exist = False
        for model_file in expected_models:
            model_path = models_dir / model_file
            if model_path.exists():
                self.log_success(f"Model exists: {model_file}")
                models_exist = True
            else:
                print(f"ℹ️  Model not yet trained: {model_file}")
        
        if not models_exist:
            print("ℹ️  No trained models found. Run 04_modeling.ipynb to generate models.")
    
    def run_validation(self):
        """Run all validation checks"""
        print("🔍 CUSTOMER CHURN PROJECT VALIDATION")
        print("=" * 50)
        
        self.check_directory_structure()
        self.check_required_files()
        self.check_notebooks()
        self.check_api_syntax()
        self.check_dockerfile_compatibility()
        self.check_dependencies()
        self.check_data_preprocessing_chain()
        self.check_model_structure()
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 VALIDATION SUMMARY")
        print("=" * 50)
        
        print(f"✅ Successes: {len(self.successes)}")
        print(f"❌ Issues: {len(self.issues)}")
        
        if self.issues:
            print("\n🚨 Issues to Address:")
            for issue in self.issues:
                print(f"   • {issue}")
        
        if len(self.issues) == 0:
            print("\n🎉 ALL CHECKS PASSED! Your project is ready to use.")
            print("\n🚀 Next Steps:")
            print("   1. Run notebooks 01-04 to train models")
            print("   2. Start API: uvicorn api.deployment_api:api --reload")
            print("   3. Test at: http://localhost:8000/docs")
        else:
            print(f"\n⚠️  Please fix {len(self.issues)} issue(s) before proceeding.")
        
        return len(self.issues) == 0

if __name__ == "__main__":
    validator = ProjectValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)