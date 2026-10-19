# AI-Powered Grocery Shopping Assistant

An egocentric, real-time multimodal AI assistant for grocery shopping. By combining wearable video (GoPro Hero 8) with open-vocabulary object detection (YOLO-World) and natural language recipe parsing, the assistant dynamically identifies required recipe ingredients on store shelves, filters transient false alarms via multi-frame temporal voting, and maintains a live shopping checklist.

---

## Team Members

- **Olina Zheng** (`oz36`)
- **Shannon Liu** (`sjl356`)
- **Paige Yun** (`phy6`)

---

## System Architecture

```
[ Recipe Text ]
       │
       ▼
[ NLP Ingredient Parser ]  ──► Extracts clean food items: ['red onion', 'garlic', 'mozzarella']
       │
       ▼
[ YOLO-World set_classes ] ──► Encodes text vocabulary into detection head once (offline)
       │
       ▼
[ GoPro Hero 8 Video Feed ]──► Streams 1080p @ 30 FPS over USB-C via GoPro Webcam app
       │
       ▼
[ Real-Time Inference ]    ──► YOLO-World on Apple Silicon MPS (Metal GPU)
       │
       ▼
[ Temporal Voting Filter ] ──► Sustained N-frame confirmation to prevent flicker
       │
       ▼
[ Shopping Checklist HUD ] ──► Live visual overlay & auditory notifications
```

---

## Project Structure

```
ai-powered-shopping/
├── README.md                  # Project overview & documentation
├── project_proposal.md        # Formal CS 4701 Milestone 2 proposal
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Files and directories ignored by Git
├── recipes/                   # Sample recipe inputs for benchmarking
├── src/
│   ├── __init__.py
│   ├── main.py                # Main application entry point
│   ├── camera.py              # Camera capture & threaded frame grabber
│   ├── recipe_parser.py       # Ingredient extraction from natural language
│   ├── detector.py            # YOLO-World open-vocabulary detector wrapper
│   ├── tracker.py             # Temporal voting & checklist state tracker
│   └── ui.py                  # OpenCV HUD checklist overlay
└── tests/                     # Unit and integration tests
```

---

## Setup & Quickstart

### Prerequisites
- macOS with Python 3.10+
- GoPro Hero 8 Black connected via USB-C with the **GoPro Webcam** desktop app running.

### Installation
```bash
git clone https://github.com/shannon-jliu/ai-powered-shopping.git
cd ai-powered-shopping
pip install -r requirements.txt
```
