# DSP Challenge: Loudspeaker Compensation Filter Design

This repository contains the implementation of a digital compensation filter designed to enhance loudspeaker audio quality. The project focuses on improving low-frequency extension and achieving a broadband linear phase response.

## 🎯 Objective
The goal is to design a **Compensator** that corrects the original system (Loudspeaker + Air + Microphone) so that the overall system response matches a specific **Desired Model**. Key requirements include:
- Correcting Non-linear Phase to **Linear Phase**.
- Enhancing magnitude response.
- Ensuring the filter remains **Stable** and **Causal**.

## 🛠️ Methodology
The project was implemented using **Python** with the following steps:

1.  **System Identification**: 
    - Used cross-correlation between `input_white_noise.wav` and `output_white_noise.wav` to identify a **128-sample bulk delay**.
    - Aligned signals to estimate the transfer function ($H_{est}$) without the bulk delay to avoid non-causality in the inverse filter.
2.  **Compensator Design**: 
    - Applied **Inverse Compensation**: $H_{comp}(f) = \frac{H_{des}(f)}{H_{est}(f)}$.
    - Evaluated the system for Stability (Minimum Phase check) and High-frequency noise amplification.
3.  **Verification**: 
    - Performed **Residual Spectrum Analysis** to ensure low error in the audible range (1kHz+).
    - Conducted a **Pre-Ringing Check** to confirm the IRF peak aligns perfectly at 128 samples without non-causal oscillations.

## 📂 File Structure
- `data/`: Contains the given input/output wav files and the desired FRF (.npy).
- `code.py`: Main script for system identification and filter generation.

## 📊 Results
| Design Method | Mean Squared Error (MSE) | Perceptual Evaluation of Audio Quality (PEAQ) |
| :--- | :--- | :--- |
| **Inverse Compensation** | **1.66E-06** | **-2.2778** |

## 💻 Requirements
- Python 3.10
- NumPy
- SciPy
- Matplotlib
- Soundfile
