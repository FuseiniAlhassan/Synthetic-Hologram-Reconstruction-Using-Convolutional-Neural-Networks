# Synthetic-Hologram-Reconstruction-Using-Convolutional-Neural-Networks

# 🔬 CNN for Holographic Object Reconstruction

This project uses a simple Convolutional Neural Network (CNN) to reconstruct object images from synthetic hologram inputs. The dataset contains paired 64x64 grayscale holograms and their corresponding object images, enabling supervised learning for image-to-image translation in computational holography.

## 📁 Dataset

The dataset includes:
- `X_train_HOLO/Labels/`: Grayscale synthetic holograms.
- `y_train_obj/objects/`: Corresponding ground truth object images.

Each image is 64x64 pixels in grayscale.

## 🧠 Model Architecture

The model consists of:
- A single convolutional layer with 20 filters to extract features.
- A transposed convolutional layer (Conv2DTranspose) to reconstruct the image from encoded features.

```python
inp = Input(shape=(64, 64, 1))
d1 = Conv2D(filters=20, kernel_size=3, strides=2, activation='relu', padding='same')(inp)
e1 = Conv2DTranspose(filters=1, kernel_size=3, strides=2, activation='relu', padding='same')(d1)
model = Model(inputs=inp, outputs=e1)

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Epochs: 200


🏋️ Training

The model was trained for 200 epochs on ~5000 image pairs. It showed stable convergence and produced smooth reconstructions of objects from their holographic representations.

Example Input and Output

Synthetic Hologram	Reconstructed Object

	


(Replace with actual images if available in repo)

📈 Results

Final training loss: ~24 MSE

Fast training (~1s per epoch on GPU)

Can generalize across various synthetic holograms


💡 Applications

Digital holography

Optical phase retrieval

Physics-informed deep learning

Image-to-image tasks in computational optics


📎 Requirements

Python 3.x

TensorFlow 2.x

NumPy, Matplotlib, PIL

Jupyter or Google Colab


Install requirements:

pip install tensorflow numpy matplotlib pillow

📄 License

MIT License

🤝 Acknowledgments

Inspired by real-world challenges in computational holography and image reconstruction in optics and photonics research.


---

Built and maintained by Alhassan Kpahambang
