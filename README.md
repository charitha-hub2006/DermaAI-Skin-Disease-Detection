# DermaAI – Skin Disease Detection Using CNN

## About the Project

DermaAI is a web-based skin disease detection application that uses a Convolutional Neural Network (CNN) to analyze uploaded skin images and predict a possible skin disease category.

The application provides a simple interface where users can upload skin images, view prediction results, and access their previous scan history.

**Note:** This project is intended for educational and research purposes only. Its predictions are not a substitute for professional medical diagnosis or advice.

## Features

- Upload skin images for analysis.
- Predict a skin disease category using a trained CNN model.
- Display prediction results through a web interface.
- Maintain scan history for registered users.
- View previously scanned images and their prediction details.
- User authentication and access to personal scan history.

## Technologies Used

- **Programming Language:** Python
- **Backend:** Flask
- **Machine Learning / Deep Learning:** Convolutional Neural Network (CNN)
- **Database:** SQLAlchemy-supported database
- **Frontend:** HTML, CSS, JavaScript, Bootstrap (where used)
- **Model Format:** Keras

## Project Structure

```text
DermaAI-Skin-Disease-Detection/
├── app.py
├── models_db.py
├── models/
├── templates/
├── static/
├── data/
├── uploads/
├── requirements.txt
└── README.md
```

*Note: The actual folders and files may vary depending on your project version.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/charitha-hub2006/DermaAI-Skin-Disease-Detection.git
cd DermaAI-Skin-Disease-Detection
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Application

Set up any required environment variables, database configuration, and model files according to the project configuration. Do not publish passwords, secret keys, or personal data.

### 5. Run the Application

```bash
python app.py
```

Open the local address shown in the terminal in your web browser. For a typical Flask development setup, this may be `http://127.0.0.1:5000/`.

## How to Use

1. Start the application.
2. Sign in or create an account if authentication is enabled.
3. Upload a skin image through the scan interface.
4. View the predicted category and any additional results displayed.
5. Open Scan History to review previous scans and their images.

## Model Information

The application uses a trained CNN model to learn visual patterns from skin images and generate predictions.

Prediction quality depends on the training dataset, image quality, class distribution, and model performance. Results should not be considered medically conclusive.

## Privacy and Security

- Do not upload private or identifiable medical images to a public repository.
- Keep database files, uploaded user images, credentials, and secret configuration files out of version control.
- Use secure configuration and authentication practices when deploying the application.

## Future Enhancements

- Improve model accuracy using suitable, diverse datasets.
- Evaluate model performance with appropriate