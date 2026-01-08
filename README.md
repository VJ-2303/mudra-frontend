# 🎭 Mudra Academy - AI-Powered Bharatanatyam Learning Platform

A beautiful, modern web application for learning Bharatanatyam mudras with real-time AI detection, interactive games, and comprehensive learning resources.

## ✨ Features

### 🤖 AI-Powered Detection
- **Real-time Mudra Recognition**: Live webcam detection using hybrid ML model
- **Rule-Based + ML Detection**: Combines geometric rules with Random Forest ML
- **One Hand Detection**: Currently supports 27+ Asamyukta Hasta mudras
- **Confidence Scoring**: Real-time feedback with accuracy metrics

### 🎮 Interactive Games
- **Quiz Game**: 40 questions about mudra theory and practice
- **Flashcard Game**: Visual mudra recognition practice
- **Sequence Game**: Drag-and-drop mudra ordering challenge

### 📚 Learning Resources
- **Complete Library**: All 28 Asamyukta Hasta mudras with images and descriptions
- **Professional UI/UX**: Beautiful, responsive design with Indian classical aesthetics
- **Mobile-Friendly**: Works seamlessly on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Modern web browser with webcam support
- 2GB free RAM (for ML model)

### Installation & Launch

1. **Clone the repository** (if not already done):
   ```bash
   cd /home/vj/Code/mudra-frontend
   ```

2. **Run the unified startup script**:
   ```bash
   ./start.sh
   ```

   This single command will:
   - ✓ Check and install all dependencies
   - ✓ Create virtual environment if needed
   - ✓ Start Flask API server (port 5000)
   - ✓ Start frontend HTTP server (port 8000)
   - ✓ Display access URLs and helpful information

3. **Open your browser** and navigate to:
   ```
   http://localhost:8000
   ```

### Stop Servers

```bash
./start.sh --stop
```

## 📁 Project Structure

```
mudra-frontend/
├── index.html                 # Landing page
├── pages/
│   ├── library.html          # Mudra library with all 28 mudras
│   ├── detection-mode.html   # Choose detection mode
│   ├── live-detection.html   # Real-time AI detection page
│   └── games/
│       ├── quiz.html         # Theory quiz game
│       ├── flashcard.html    # Visual recognition game
│       └── sequence.html     # Mudra ordering game
├── js/
│   └── detection-engine.js   # Webcam capture & API client
├── css/                       # Styles
├── assets/
│   └── images/
│       └── Mudras/           # 27 mudra images
├── ml/
│   ├── hybrid_webcam.py      # Core ML detection logic
│   ├── api_server.py         # Flask API wrapper
│   ├── mudra_rf_model.pkl    # Trained ML model
│   └── requirements.txt      # Python dependencies
└── start.sh                   # Unified startup script
```

## 🔧 Technical Architecture

### Backend (Flask API)
- **Framework**: Flask + Flask-CORS
- **ML Model**: Random Forest (scikit-learn)
- **Hand Detection**: MediaPipe Hands
- **Detection Logic**: Hybrid approach from `hybrid_webcam.py`
  - Rule-based detection (strict geometric patterns)
  - ML fallback (probabilistic classification)
  - FSM for flicker-free output

### Frontend (Vanilla JS)
- **Framework**: None (Pure vanilla JavaScript)
- **Webcam**: WebRTC getUserMedia API
- **Communication**: REST API (JSON)
- **Frame Rate**: 6-7 FPS for optimal performance
- **Image Quality**: 0.7 JPEG compression for efficient transfer

### Optimization Features
- Frame capture interval: 150ms (reduces API calls)
- Base64 image compression: 70% quality
- Stateless API design: No server-side session state
- Client-side statistics tracking
- Efficient canvas-based frame capture

## 🎯 Supported Mudras

The system currently detects **27 Asamyukta Hasta mudras**:

**Rule-Based Detection** (16 mudras):
- Pataka, Tripataka, Ardhapataka, Kartari Mukham
- Mayura, Arala, Shukatunda, Mushti
- Shikhara, Kapitta, Suchi, Chandrakala
- Padmakosha, Sarpashirsha, Mrigashirsha, Simhamukha
- And more...

**ML Detection** (All mudras):
- Complete coverage including ambiguous hand gestures
- Confidence threshold: 55%
- 17 geometric features extracted per frame

## 🌐 API Endpoints

### GET `/health`
Check server status and model info
```json
{
  "status": "ok",
  "model_loaded": true,
  "mudra_count": 27,
  "rule_based_mudras": 16,
  "ml_mudras": 27
}
```

### POST `/detect`
Detect mudra from base64 image
```json
{
  "image": "data:image/jpeg;base64,..."
}
```

Response:
```json
{
  "success": true,
  "hand_detected": true,
  "mudra": "Pataka Mudra",
  "confidence": 1.0,
  "method": "RULE"
}
```

### GET `/mudras`
List all supported mudras

## 📊 Performance

- **Detection Latency**: < 50ms per frame
- **Frame Rate**: 6-7 FPS (client-side capture)
- **API Response Time**: 30-80ms average
- **Accuracy**: 
  - Rule-based: 100% when conditions met
  - ML-based: ~85% with confidence > 0.55

## 🎨 Design System

- **Color Palette**:
  - Primary: Maroon (#8B2942)
  - Secondary: Gold (#D4A84B)
  - Background: Cream (#FFF8F0)
- **Typography**:
  - Headings: Playfair Display (serif)
  - Body: Inter (sans-serif)
- **Style**: Indian classical meets modern minimalism

## 🐛 Troubleshooting

### Camera not working
- Ensure browser has camera permissions
- Check if camera is already in use by another app
- Try in Chrome/Firefox (better WebRTC support)

### API server not starting
- Check if port 5000 is available: `lsof -ti:5000`
- View logs: `tail -f api_server.log`
- Ensure virtual environment is activated

### Detection not working
- Verify API server is running: `curl http://localhost:5000/health`
- Check browser console for errors
- Ensure good lighting and clear hand visibility

## 🚧 Roadmap

- [ ] Two-hand detection (Samyukta Hasta mudras)
- [ ] Progress tracking and user accounts
- [ ] Video tutorials for each mudra
- [ ] Mobile app (React Native)
- [ ] Offline mode with TensorFlow.js

## 👥 Contributing

This is a hackathon project. Contributions welcome!

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- MediaPipe for hand landmark detection
- scikit-learn for ML framework
- Google Fonts for typography
- Bharatanatyam practitioners for mudra guidance

---

**Made with ❤️ for preserving and promoting Indian classical dance through technology**

*Last updated: January 7, 2026*
