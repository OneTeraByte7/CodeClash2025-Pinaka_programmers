----
----
----
# 🚗 Collision Detection AI - CodeClash2025 🚀

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Repo stars](https://img.shields.io/github/stars/OneTeraByte7/CodeClash2025-Pinaka_programmers?style=social)](https://github.com/OneTeraByte7/CodeClash2025-Pinaka_programmers/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/OneTeraByte7/CodeClash2025-Pinaka_programmers?style=social)](https://github.com/OneTeraByte7/CodeClash2025-Pinaka_programmers/network/members)

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Technologies Used](#-technologies-used)
- [Installation Guide](#-installation-guide)
- [Usage](#-usage)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

This AI-powered **Collision Detection System** addresses the critical challenge of real-world vehicle accidents, which are often caused by inefficient sensors and insufficient training data. By leveraging **custom-built machine learning models** trained on a **self-generated, high-fidelity dataset**, this project provides a robust solution for predicting and preventing collisions in real-time.

The system analyzes vehicle dynamics—such as velocity, speed, and angle—along with other risk factors to deliver timely warnings and enhance driver safety.

---

## ✨ Key Features

*   **Live Collision Detection**: Utilizes OpenCV for real-time video capture from webcams or video files to predict potential crashes by analyzing vehicle telemetry.
*   **Lane-Based Collision Model**: Implements a specialized model to detect and warn about risks associated with lane departures and unsafe lane changes.
*   **High-Accuracy Custom Dataset**: Trained on a unique, self-generated dataset that mirrors real-world driving scenarios, leading to more reliable predictions.
*   **Post-Ride Analysis**: Offers a comprehensive ride report after each session, including detailed accuracy calculations and performance metrics.
*   **Persistent Model Training**: Saves trained models as `.pkl` files, allowing for continuous learning and improvement over time without retraining from scratch.
*   **Real-time Video Processing**: Employs `OpenCV` for efficient video rendering, object tracking, and data visualization.

---

## ⚙️ How It Works

The system follows a multi-stage process to ensure accurate collision detection:

1.  **Data Generation**: A custom dataset is generated to simulate various driving scenarios, capturing key metrics like speed, angle, and proximity to other vehicles.
2.  **Model Training**: Machine learning models (from Scikit-Learn and TensorFlow) are trained on this dataset to learn the patterns that precede a collision. The trained models are serialized and saved as `.pkl` files.
3.  **Real-time Video Input**: The system captures video from a live webcam or a pre-recorded file using OpenCV.
4.  **Object Detection & Tracking**: Vehicles and lanes are identified and tracked across frames.
5.  **Feature Extraction**: For each detected vehicle, real-time data (velocity, angle, etc.) is extracted.
6.  **Prediction**: The trained model processes these features to calculate a real-time risk score and predict the likelihood of a collision.
7.  **Alerting & Visualization**: The system provides on-screen alerts and visualizes the tracking and risk assessment on the video feed.

---

## 🛠️ Technologies Used

-   **Core Language**: **Python** (3.8+)
-   **Machine Learning**: **Scikit-Learn**, **TensorFlow/Keras**
-   **Data Handling**: **Pandas**, **NumPy**
-   **Video Processing**: **OpenCV**
-   **Data Visualization**: **Matplotlib**, **Seaborn**

---

## 🚀 Installation Guide

Follow these steps to set up the project on your local machine.

1️⃣ **Clone the Repository**
```bash
git clone https://github.com/OneTeraByte7/CodeClash2025-Pinaka_programmers.git
cd CodeClash2025-Pinaka_programmers
```

2️⃣ **Create and Activate a Virtual Environment**
```bash
# For macOS/Linux
python3 -m venv env
source env/bin/activate

# For Windows
python -m venv env
.\env\Scripts\activate
```

3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

4️⃣ **Verify Installation**
> After installation, you can run the main script to ensure everything is set up correctly.

---

## ▶️ Usage

To run the collision detection system, execute the main script from the root directory of the project:

```bash
python main.py
```

The application will start, and you can choose to use a live webcam feed or a video file for detection.

---

## 🖼️ Screenshots

| Main Interface | Live Detection |
| :---: | :---: |
| !Main UI | !Live Detection |

| Post-Ride Analysis | Model Accuracy |
| :---: | :---: |
| !Ride Report | !Accuracy Plot |

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements or want to fix a bug, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/YourFeature`).
5.  Open a Pull Request.

Please make sure to update tests as appropriate.

---

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

