import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const AQI_LABELS = {
  0: { label: "Good", color: "#00e400" },
  1: { label: "Moderate", color: "#ffff00" },
  2: { label: "Unhealthy (Sensitive)", color: "#ff7e00" },
  3: { label: "Unhealthy", color: "#ff0000" },
  4: { label: "Very Unhealthy", color: "#8f3f97" },
  5: { label: "Severe", color: "#7e0023" }
};

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [params, setParams] = useState({
    Year: 2026, Month: 2, Day: 2, Hour: 10,
    AQI: 50, PM25: 12.0, PM10: 20.0, O3: 35.0, CO: 0.5, SO2: 8.0, NO2: 15.0
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setParams({ ...params, [name]: parseFloat(value) || 0 });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    const formData = new FormData();
    formData.append('image', file);
    formData.append('params', JSON.stringify(params));

    try {
      const res = await axios.post('http://127.0.0.1:5001/predict', formData);
      setPrediction(res.data);
    } catch (err) {
      alert("Error connecting to AI Server. Ensure Flask is running on port 5000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-container">
      <header className="header">
        <h1>AMAYRAH <span className="ai-badge">Multimodal AI</span></h1>
        <p>Satellite Image & Meteorological Data Fusion for AQI Analysis</p>
      </header>

      <div className="main-content">
        <form onSubmit={handleSubmit} className="analysis-form">
          <section className="upload-section">
            <h3>1. Visual Input</h3>
            <div className={`drop-zone ${preview ? 'has-image' : ''}`}>
              {preview ? <img src={preview} alt="Preview" /> : <p>Drag & Drop Satellite Image</p>}
              <input type="file" onChange={(e) => setFile(e.target.files[0])} required />
            </div>
          </section>

          <section className="params-section">
            <h3>2. Environmental Parameters</h3>
            <div className="params-grid">
              {Object.keys(params).map(key => (
                <div className="input-group" key={key}>
                  <label>{key}</label>
                  <input type="number" name={key} value={params[key]} onChange={handleInputChange} step="any" />
                </div>
              ))}
            </div>
          </section>

          <button type="submit" className="predict-btn" disabled={loading || !file}>
            {loading ? <div className="spinner"></div> : "Analyze Atmospheric Conditions"}
          </button>
        </form>

        <section className="result-panel">
          <h3>3. AI Diagnostic Result</h3>
          {prediction ? (
            <div className="prediction-card">
              <div className="aqi-meter" style={{ borderColor: AQI_LABELS[prediction.prediction]?.color }}>
                <span className="aqi-value">{AQI_LABELS[prediction.prediction]?.label}</span>
                <span className="aqi-subtitle">Class {prediction.prediction}</span>
              </div>
              <div className="confidence-bar-container">
                <label>AI Confidence Score</label>
                <div className="bar-bg">
                  <div className="bar-fill" style={{ width: `${prediction.confidence * 100}%` }}></div>
                </div>
                <span>{(prediction.confidence * 100).toFixed(1)}%</span>
              </div>
            </div>
          ) : (
            <div className="empty-state">Waiting for data input...</div>
          )}
        </section>
      </div>
    </div>
  );
}

export default App;