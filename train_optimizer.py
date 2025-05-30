#!/usr/bin/env python3
"""
Train the Credit Score Optimizer
"""

from pathlib import Path
import kagglehub
from credit_score_optimizer import CreditScoreOptimizer

def main():
    print("🚀 Training Credit Score Optimizer\n")
    
    # Check/download dataset
    dataset_path = Path("data/train.csv")
    
    if not dataset_path.exists():
        print("📥 Downloading dataset from Kaggle...")
        try:
            path = kagglehub.dataset_download("parisrohan/credit-score-classification")
            import shutil
            Path("data").mkdir(exist_ok=True)
            
            for file in Path(path).rglob("*.csv"):
                if "train" in file.name.lower():
                    shutil.copy(file, dataset_path)
                    print(f"✅ Dataset saved to {dataset_path}")
                    break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please download manually from Kaggle")
            return
    
    # Train optimizer
    optimizer = CreditScoreOptimizer()
    optimizer.train(dataset_path)
    
    # Save model
    optimizer.save_model()
    
    print("\n✅ Training complete! You can now run app.py")

if __name__ == "__main__":
    main()