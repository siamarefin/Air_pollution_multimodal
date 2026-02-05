/**
 * Air Pollution Prediction Server
 * Simple Node.js/Express server that calls Python backend
 */

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 5001;

// Middleware
app.use(cors());
app.use(express.json());

// Configure multer for file uploads
const upload = multer({
  dest: 'uploads/',
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// Model configuration
const MODEL_PATH = '/Users/siamarefin/Amayrah/Air polution/model /best_base.pth';

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    model_path: MODEL_PATH,
    timestamp: new Date().toISOString()
  });
});

// Get class names
app.get('/classes', (req, res) => {
  res.json({
    classes: [
      'a_Good',
      'b_Satisfactory',
      'c_Moderate',
      'd_Poor',
      'e_VeryPoor',
      'f_Severe'
    ],
    num_classes: 6
  });
});

// Prediction endpoint
app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    // Validate inputs
    if (!req.file) {
      return res.status(400).json({ error: 'No image provided' });
    }

    if (!req.body.params) {
      return res.status(400).json({ error: 'No parameters provided' });
    }

    const params = JSON.parse(req.body.params);
    const imagePath = req.file.path;

    // Call Python prediction script
    const pythonProcess = spawn('python3', [
      'predict.py',
      MODEL_PATH,
      imagePath,
      JSON.stringify(params)
    ]);

    let result = '';
    let error = '';

    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });

    pythonProcess.on('close', (code) => {
      // Clean up uploaded file
      fs.unlink(imagePath, (err) => {
        if (err) console.error('Error deleting temp file:', err);
      });

      if (code !== 0) {
        console.error('Python error:', error);
        return res.status(500).json({
          error: 'Prediction failed',
          details: error
        });
      }

      try {
        const prediction = JSON.parse(result);
        res.json(prediction);
      } catch (e) {
        res.status(500).json({
          error: 'Failed to parse prediction result',
          details: result
        });
      }
    });

  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({
      error: 'Internal server error',
      details: error.message
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log('='.repeat(80));
  console.log('AIR POLLUTION PREDICTION SERVER');
  console.log('='.repeat(80));
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`Model path: ${MODEL_PATH}`);
  console.log('Endpoints:');
  console.log(`  GET  /health  - Health check`);
  console.log(`  GET  /classes - Get class names`);
  console.log(`  POST /predict - Make prediction`);
  console.log('='.repeat(80));
});