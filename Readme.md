# Air Pollution Server - Quick Setup Guide

## 📦 Files You Need

```
your-project/
├── server.js              # Node.js server (port 5001)
├── predict.py             # Python prediction script
├── package.json           # Node.js dependencies
├── uploads/               # Auto-created for temp files
└── model/
    └── best_base.pth     # Your trained model
```

## 🚀 Quick Start (2 Steps)

### Step 1: Install Dependencies

```bash
# Install Node.js packages
npm install

# Install Python packages
pip install torch torchvision transformers opencv-python albumentations numpy
```

### Step 2: Run Server

```bash
# Update model path in server.js (line 21)
# Then start server:
node server.js
```

Server runs on: **http://localhost:5001**

## 📝 Update Model Path

In `server.js`, line 21:
```javascript
const MODEL_PATH = '/Users/siamarefin/Amayrah/Air polution/model/best_base.pth';
```

Change to your actual model path.

## 🧪 Test the Server

### Test 1: Health Check
```bash
curl http://localhost:5001/health
```

### Test 2: Get Classes
```bash
curl http://localhost:5001/classes
```

### Test 3: Make Prediction
```bash
curl -X POST http://localhost:5001/predict \
  -F "image=@test_image.jpg" \
  -F 'params={"Year":2026,"Month":2,"Day":2,"Hour":10,"AQI":50,"PM25":12.0,"PM10":20.0,"O3":35.0,"CO":0.5,"SO2":8.0,"NO2":15.0}'
```

## 📊 API Endpoints

### GET /health
Returns server status

**Response:**
```json
{
  "status": "healthy",
  "model_path": "/path/to/model.pth",
  "timestamp": "2024-02-02T10:00:00.000Z"
}
```

### GET /classes
Returns available class names

**Response:**
```json
{
  "classes": ["a_Good", "b_Satisfactory", "c_Moderate", "d_Poor", "e_VeryPoor", "f_Severe"],
  "num_classes": 6
}
```

### POST /predict
Make a prediction

**Request:**
- `image`: Image file (multipart/form-data)
- `params`: JSON string with parameters

**Parameters:**
```json
{
  "Year": 2026,
  "Month": 2,
  "Day": 2,
  "Hour": 10,
  "AQI": 50,
  "PM25": 12.0,
  "PM10": 20.0,
  "O3": 35.0,
  "CO": 0.5,
  "SO2": 8.0,
  "NO2": 15.0
}
```

**Response:**
```json
{
  "prediction": 0,
  "class_name": "a_Good",
  "confidence": 0.95,
  "probabilities": {
    "0": 0.95,
    "1": 0.03,
    "2": 0.01,
    "3": 0.005,
    "4": 0.003,
    "5": 0.002
  },
  "input_params": {...}
}
```

## 🔧 Troubleshooting

### Server won't start
- Check if port 5001 is available: `lsof -i :5001`
- Kill process if needed: `kill -9 <PID>`

### Python errors
- Ensure Python 3.7+ is installed: `python3 --version`
- Install all dependencies: `pip install -r backend_requirements.txt`

### Model not loading
- Check model path is correct
- Ensure `best_base.pth` file exists
- Verify file permissions

### Image upload fails
- Check file size (max 10MB)
- Create uploads directory: `mkdir uploads`
- Check file permissions

## 🎯 Using with React Frontend

Your React `App.js` already points to port 5001:
```javascript
axios.post('http://127.0.0.1:5001/predict', formData)
```

Just start both:
1. `node server.js` (Terminal 1)
2. `npm start` in React folder (Terminal 2)

## 📋 Complete Example

```bash
# Terminal 1: Start server
cd server-folder
npm install
node server.js

# Terminal 2: Start React
cd react-folder
npm start

# Browser opens at http://localhost:3000
# Upload image + enter params + click predict
```

## ✅ Checklist

Before running:
- [ ] Node.js installed (v14+)
- [ ] Python 3.7+ installed
- [ ] All npm packages installed (`npm install`)
- [ ] All pip packages installed
- [ ] Model path updated in server.js
- [ ] Model file exists at path
- [ ] Port 5001 is free

## 🎉 Success!

If server starts successfully, you'll see:
```
================================================================================
AIR POLLUTION PREDICTION SERVER
================================================================================
Server running on http://localhost:5001
Model path: /Users/siamarefin/Amayrah/Air polution/model/best_base.pth
Endpoints:
  GET  /health  - Health check
  GET  /classes - Get class names
  POST /predict - Make prediction
================================================================================
```

Your server is ready! 🚀