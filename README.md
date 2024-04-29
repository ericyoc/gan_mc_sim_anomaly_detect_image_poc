# Anomaly Detection using GAN and Monte Carlo Simulation

This Python code demonstrates the use of a Generative Adversarial Network (GAN) and Monte Carlo simulation for anomaly detection on the MNIST dataset. The combination of GANs and Monte Carlo simulation provides a powerful approach for identifying anomalies in complex datasets.

## Overview

Anomaly detection is the process of identifying instances that deviate significantly from the norm. In this code, we train a GAN to learn the normal patterns in the MNIST dataset, which consists of handwritten digit images. The trained GAN is then used to generate synthetic images, and a discriminator model is employed to assign anomaly scores to both real and generated images.

## GAN Architecture

The GAN consists of two main components: a generator and a discriminator.

- The generator takes random noise as input and learns to generate realistic-looking images that resemble the training data.
- The discriminator takes an image as input and learns to distinguish between real and generated images.

During training, the generator and discriminator are trained simultaneously in a competitive manner. The generator aims to fool the discriminator by generating realistic images, while the discriminator aims to correctly classify real and generated images.

## Monte Carlo Simulation

After training the GAN, we perform anomaly detection using Monte Carlo simulation. The steps involved are as follows:

1. Generate a large number of synthetic images using the trained generator.
2. Use the trained discriminator to assign anomaly scores to each generated image.
3. Calculate a threshold based on the distribution of anomaly scores (e.g., 95th percentile).
4. Classify test images as anomalies if their anomaly scores fall below the threshold.

The Monte Carlo simulation allows us to estimate the distribution of anomaly scores for normal instances. By setting a threshold based on this distribution, we can identify anomalies that deviate significantly from the norm.

## Importance of Combining GAN and Monte Carlo Simulation

The combination of GANs and Monte Carlo simulation offers several advantages for anomaly detection:

1. GANs learn the underlying patterns and distributions of normal data, enabling them to generate realistic samples that capture the essential characteristics of the dataset.
2. Monte Carlo simulation provides a probabilistic approach to estimate the distribution of anomaly scores for normal instances. This allows us to set a threshold for anomaly detection based on the expected behavior of normal data.
3. By generating a large number of synthetic samples using the GAN, we can obtain a more robust estimate of the anomaly score distribution, reducing the impact of individual outliers or noise.
4. The combination of GANs and Monte Carlo simulation can handle complex and high-dimensional datasets, making it suitable for a wide range of anomaly detection tasks.

## Usage

To run the code, ensure you have the required dependencies installed (TensorFlow, Keras, matplotlib). The code assumes the availability of a GPU for training, but you can modify it to run on CPU if needed.

The code will train the GAN on the MNIST dataset, perform anomaly detection using Monte Carlo simulation, and display the following results:

- Plots of an anomaly test image and a normal test image
- Histogram of the Monte Carlo simulation results, showing the distribution of anomaly scores
- Summary of the results, including the threshold, mean anomaly score, minimum and maximum anomaly scores, and an interpretation of the scores
- Descriptions of the anomaly test image and the normal test image

Feel free to explore and modify the code to adapt it to your specific anomaly detection tasks or datasets.

## Acknowledgments

This code is inspired by the concepts of Generative Adversarial Networks and Monte Carlo simulation for anomaly detection. It builds upon the MNIST dataset and utilizes the TensorFlow and Keras libraries for implementation.

## License

This code is released under the [MIT License](LICENSE).
