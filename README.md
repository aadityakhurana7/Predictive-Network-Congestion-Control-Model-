# 🧠 Predictive Network Congestion Model  

## 📌 Overview  
This project implements a **Predictive Network Congestion Detection System** using **Machine Learning** and **IoT-based network monitoring**.  
It analyzes live network traffic captured via **TShark**, extracts key metrics, and predicts congestion status using a **Random Forest model**, visualized through a **Streamlit dashboard**.  

---

## ⚙️ Features  
- 📡 Real-time traffic capture via TShark  
- 🧩 Data preprocessing and feature engineering  
- 🌲 Random Forest model for congestion prediction  
- 📊 Interactive Streamlit web app for visualization and live inference  
- 💾 Pre-trained model (`.pkl`) for quick deployment  

---

## 🧰 Tech Stack  
- **Python 3.x**  
- **Streamlit**  
- **Scikit-learn**  
- **Pandas**, **NumPy**  
- **TShark / PyShark**  

---

## 📁 Project Structure  
```
├── app.py                       # Streamlit app for prediction and UI  
├── data.py                      # Data processing and feature extraction  
├── python.py / test.py           # Model training and testing scripts  
├── network_congestion_data.csv   # Dataset used for training  
├── network_congestion_model.pkl  # Trained RandomForest model  
├── mathsyn.docx                  # Supporting documentation  
```

---

Then open the local URL displayed in your terminal (e.g., `http://localhost:8501`).  

---

## 📈 Model Details  
- **Algorithm:** Random Forest Classifier  
- **Inputs:** Packet loss, latency, jitter, queue size, throughput  
- **Output:** Network status → `Normal` / `Congested`  

---

## 🧪 Use Case  
This system helps detect and predict network congestion before it impacts performance — useful for **data centers**, **IoT systems**, and **enterprise networks**.  

---

## 🔮 Future Scope  
- Integrate deep learning models (LSTM, ANN)  
- Enable live monitoring via APIs  
- Extend to distributed cloud systems  

---

