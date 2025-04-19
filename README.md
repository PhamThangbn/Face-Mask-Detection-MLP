# 😷 Face Mask Detection using MLP

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A Python-based project utilizing a **Multilayer Perceptron (MLP)** neural network to detect faces and classify whether individuals are **wearing a face mask** or **not** from **images**, **videos**, **webcam**, or **live camera feed**. The application features a user-friendly **GUI** built with **Tkinter** and integrates powerful libraries such as **OpenCV**, **scikit-learn**, **MTCNN**, **Pillow**, and more.

---

## 🚀 Getting Started

Follow these steps to set up and run the project:

### 1. Clone the Repository

Clone the project to your local machine:

```bash
git clone https://github.com/PhamThangbn/Face-Mask-Detection-MLP.git
cd Face-Mask-Detection-MLP
```

### 2. Install Required Libraries

Install the necessary Python libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)

If you want to train a new MLP model, run the training script:

```bash
python train.py
```

The trained model will be saved as:

```
models/Model_MLP_new.pkl
```

⚠️ **Note**:

- The newly generated `.pkl` file **will not overwrite** the existing model.
- To use the new model as the default, rename it manually (e.g., to `train_MLP_new.pkl`) and update the model path in `main.py` if needed.

### 4. Run the Main Application

Launch the face mask detection GUI:

```bash
python main.py
```

---

## 📂 Project Structure

Below is the directory structure of the project:

```
Face-Mask-Detection-MLP/
│
├── dataset/                # Contains categorized face image data
│   ├── with_mask/          # Images of people wearing masks
│   └── without_mask/       # Images of people without masks
│
├── models/                 # Stores trained model files
│   └── *.pkl               # Pre-trained or newly trained MLP models in pickle format
│
├── output_videos/          # Contains videos after processing with annotations
│
├── main.py                 # Main script to launch the face detection GUI
│
├── train.py                # Script to train the MLP model from dataset
│
├── requirements.txt        # Required Python libraries for this project
│
└── README.md               # This documentation file
```

---

## 🛠️ Requirements

The project depends on the following Python libraries (listed in `requirements.txt`):

| Library       | Purpose                            |
| ------------- | ---------------------------------- |
| opencv-python | Image and video processing         |
| numpy         | Numerical computations             |
| joblib        | Loading the trained MLP model      |
| Pillow        | Image handling in Tkinter GUI      |
| mtcnn         | Face detection using MTCNN model   |
| tensorflow    | Deep learning operations for MTCNN |

Install them using:

```bash
pip install -r requirements.txt
```

---

## 🎯 Features

- **Face Detection**: Uses **MTCNN** to accurately detect faces in images or video frames.
- **Mask Classification**: Classifies whether a person is **wearing a mask** or **not** using a pre-trained **MLP** model.
- **GUI Interface**: A **Tkinter-based GUI** for easy interaction, supporting:
  - Image processing with navigation (`<`, `<current/total>`, `>`).
  - Video and webcam processing with real-time annotations.
- **Output Storage**: Processed videos are saved with annotations in the `output_videos/` directory.
- **Flexible Input**: Supports **images**, **videos**, **webcam**, or **live camera feed**.

---

## ⚙️ How It Works

1. **Face Detection**:

   - The **MTCNN** model detects faces in input images or video frames with high accuracy.
   - Detected faces are cropped and processed for classification.

2. **Mask Classification**:

   - The cropped face is resized to `128x128`, normalized, and fed into the **MLP** model.
   - The model outputs whether the person is **wearing a mask** or **not**.

3. **GUI Interaction**:
   - **Images**: Select multiple images to process, navigate through results using `<` and `>` buttons.
   - **Videos/Webcam**: Process video files or live webcam feed, with results displayed in real-time and saved to `output_videos/`.
   - **Stop Button**: Pause processing at any time to review results or switch functions.

---

## 📊 Results

The model achieves an accuracy of **92%** on the test set.

Below is the results of the code for predicting the images:

![image](https://github.com/PhamThangbn/Face-Mask-Detection-MLP/blob/main/images/with_mask.JPG)

![image](https://github.com/PhamThangbn/Face-Mask-Detection-MLP/blob/main/images/without_mask.JPG)

---

## 📝 Notes

- **Model Path**: Ensure the model file (`models/train_MLP_new.pkl`) exists or update the path in `main.py` if using a custom model.
- **Performance**: MTCNN may be slower for real-time video/webcam due to its deep learning nature. For better performance, consider using a GPU or resizing input frames.
- **Dataset**: The `dataset/` folder contains sample images (`with_mask/` and `without_mask/`) for training. Add more data to improve model accuracy.

---

## 🙏 Acknowledgments

Special thanks to the following contributors for their valuable support and collaboration:

- **Pham Van Thang**
- **Nguyen Duc Tam**
- **Ngo Sach Tien**
- **Bui Sy Toan**

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

---

## 📧 Contact

For questions or feedback, reach out to:

- **Author**: Pham Thang
- **GitHub**: [PhamThangbn](https://github.com/PhamThangbn)

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
