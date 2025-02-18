# Roman-Urdu-Poetry-Generator-LSTM

This repository contains two Python scripts for training and generating Roman Urdu poetry using an LSTM model.

## Project Overview
The project is divided into two parts:
1. **Training the Model (`train_poetry_model.py`)**: This script trains an LSTM-based model on Roman Urdu poetry and saves the trained model along with the tokenizer.
2. **Generating Poetry (`generate_poetry.py`)**: This script loads the trained model and tokenizer to generate poetry based on user input via a Gradio interface.

---

## Installation & Setup

### Prerequisites
Make sure you have Python installed (>=3.8). Install dependencies using:
```bash
pip install -r requirements.txt
```

### Dataset
Ensure you have the poetry dataset (`Roman-Urdu-Poetry.csv`) in the same directory as `train_poetry_model.py` before running the training script.

---

## Training the Model
To train the LSTM model on Roman Urdu poetry, run:
```bash
python train_poetry_model.py
```
This will:
- Load and preprocess the dataset
- Train the model using an LSTM-based architecture
- Save the trained model and tokenizer as `poetry_lstm_model.h5` and `tokenizer.pkl`

---

## Generating Poetry
Once the model is trained and saved, you can generate poetry by running:
```bash
python generate_poetry.py
```
This will launch a **Gradio-based web interface**, where you can:
- Enter a **seed word**
- Choose the **number of words to generate**
- Adjust the **temperature** (creativity level)

---

## File Structure
```
├── Roman-Urdu-Poetry.csv        # Poetry dataset
├── train_poetry_model.py        # Model training script
├── generate_poetry.py           # Model loading & poetry generation script
├── requirements.txt             # Python dependencies
├── poetry_lstm_model.h5         # Saved LSTM model (after training)
├── tokenizer.pkl                # Saved tokenizer
├── README.md                    # Project documentation
```

---

## Dependencies
The project uses:
- `TensorFlow` for training the LSTM model
- `Gradio` for an interactive poetry generator
- `NumPy`, `Pandas`, and `Matplotlib` for preprocessing and visualization

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## Future Improvements
- Improve the dataset by adding more poetry samples
- Optimize model hyperparameters for better performance
- Deploy the Gradio interface as a web app

---

## Contributing
Feel free to contribute to this project by submitting issues or pull requests!

---

## License
This project is licensed under the MIT License.

![Uploading image.png…]()

