#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Voice Phishing Detection - Full Pipeline${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Activate virtual environment
echo -e "${GREEN}[1/8] Activating virtual environment...${NC}"
source venv/bin/activate

# Step 1: Data Exploration
echo -e "\n${GREEN}[2/8] Exploring dataset...${NC}"
python src/01_data_exploration.py

# Step 2: Data Augmentation
echo -e "\n${GREEN}[3/8] Augmenting data...${NC}"
python src/02_data_augmentation.py

# Step 3: Feature Extraction
echo -e "\n${GREEN}[4/8] Extracting features...${NC}"
python src/03_feature_extraction.py

# Step 4: Train DNN
echo -e "\n${GREEN}[5/8] Training DNN model...${NC}"
python src/04_train_dnn.py

# Step 5: Train XGBoost
echo -e "\n${GREEN}[6/8] Training XGBoost model...${NC}"
python src/05_train_xgboost.py

# Step 6: Ensemble Evaluation
echo -e "\n${GREEN}[7/8] Evaluating ensemble...${NC}"
python src/06_ensemble_eval.py

# Step 7: Start Flask App
echo -e "\n${GREEN}[8/8] Starting Flask application...${NC}"
echo -e "${BLUE}Access the application at: http://localhost:5000${NC}\n"
python app/flask_app.py
