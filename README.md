
# Model Training Project

This project is designed to train a model using the provided `train_model.py` script. Follow the step-by-step instructions below to set up the environment, install dependencies, and execute the training script.

---

## Prerequisites

- Python 3.8 or later installed on your system.
- `pip` (Python's package manager) installed.

---

## Setup Instructions

### 1. Create and Activate a Virtual Environment

#### On macOS/Linux:
```bash
python3 -m venv venv  # Create the virtual environment
source venv/bin/activate  # Activate the virtual environment
```

#### On Windows:
```bash
python -m venv venv  # Create the virtual environment
venv\Scripts\activate  # Activate the virtual environment
```

### 2. Upgrade `pip`

Ensure `pip` is up-to-date:
```bash
pip install --upgrade pip
```

### 3. Install Required Libraries

Install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---
## Dataset Instructions

1. **Download the Dataset**
   - Download the dataset from the following link: [PEDRo Dataset](https://zenodo.org/records/13331985)

2. **Extract the Dataset**
   - Extract the downloaded dataset into a directory named `PEDro`.

3. **Generate Additional Files**
   - Run the `generate_script.py` file to preprocess or generate necessary files.
     ```bash
     python generate_script.py
     ```

Ensure the `PEDro` directory is correctly structured before proceeding to the next steps.
---

## Running the Training Script

Once the environment is set up and dependencies are installed, you can run the training script using the following command:

```bash
python src/train_model.py
```

---

## Generating a `requirements.txt` File (Optional)

If you need to generate a `requirements.txt` file from an existing project environment, activate the virtual environment and run:
```bash
pip freeze > requirements.txt
```

---

## Notes

- Always activate the virtual environment before running any commands to ensure the correct dependencies are used.
- If you encounter any issues with dependencies, recreate the environment using the instructions provided above.

---

## License

This project is distributed under End-User License Agreement (EULA).
