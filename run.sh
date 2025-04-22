#!/bin/bash

# Shell script to run the explainable AI credit scoring project

# Set up Python environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Parse command line arguments
GENERATE_DATA=false
TRAIN_MODELS=false
OPTIMIZE=false
RUN_APP=false
ALL=false

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --generate-data)
        GENERATE_DATA=true
        shift
        ;;
        --train-models)
        TRAIN_MODELS=true
        shift
        ;;
        --optimize)
        OPTIMIZE=true
        shift
        ;;
        --run-app)
        RUN_APP=true
        shift
        ;;
        --all)
        ALL=true
        shift
        ;;
        *)
        # Unknown option
        shift
        ;;
    esac
done

# Run the selected steps or the complete pipeline
if $ALL; then
    echo "Running complete pipeline..."
    python main.py --all
elif [ $GENERATE_DATA = true ] || [ $TRAIN_MODELS = true ] || [ $RUN_APP = true ]; then
    COMMAND="python main.py"
    
    if [ $GENERATE_DATA = true ]; then
        COMMAND="$COMMAND --generate-data"
    fi
    
    if [ $TRAIN_MODELS = true ]; then
        COMMAND="$COMMAND --train-models"
    fi
    
    if [ $OPTIMIZE = true ]; then
        COMMAND="$COMMAND --optimize"
    fi
    
    if [ $RUN_APP = true ]; then
        COMMAND="$COMMAND --run-app"
    fi
    
    echo "Running: $COMMAND"
    eval $COMMAND
else
    # Default behavior: just run the app with the example model and data
    echo "No options specified. Running the web application..."
    python main.py --run-app
fi

# Deactivate virtual environment
deactivate 