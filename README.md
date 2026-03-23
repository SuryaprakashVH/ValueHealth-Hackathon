# LexGuard: AI-Powered Multi-Agent Contract Review System

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

LexGuard is an intelligent contract analysis platform that leverages multiple AI agents to automate the review of legal documents. Built with FastAPI, Streamlit, and Groq AI, it provides comprehensive risk assessment, clause comparison, and automated report generation for PDF contracts.

## 🚀 Features

### Core Capabilities
- **Automated PDF Analysis**: Processes legal contracts (PDFs) with support for both text-based and scanned documents
- **Multi-Agent Pipeline**: 5 specialized AI agents working in coordination:
  - Document Ingestion Agent: Text extraction and contract type detection
  - Metadata Extraction Agent: Identifies parties, dates, and key terms
  - Clause Comparison Agent: Compares clauses against standard templates
  - Risk Classification Agent: Analyzes and classifies legal risks
  - Report Generation Agent: Creates detailed PDF reports

### User Interfaces
- **Web Application**: Streamlit-based UI for file upload and real-time analysis
- **REST API**: FastAPI backend for programmatic access
- **Chatbot Interface**: Interactive AI assistant for contract queries

### Advanced Features
- **Contract Type Recognition**: Automatically detects NDA, SLA, MSA, and other contract types
- **Deviation Detection**: Identifies deviations from standard clause templates
- **Risk Assessment**: Severity-based risk classification (HIGH/MEDIUM/LOW)
- **Report Generation**: Professional PDF reports with executive summaries
- **Database Integration**: MongoDB storage for review history and analytics

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI API   │    │   Chatbot UI    │
│                 │    │                 │    │                 │
│ • File Upload   │    │ • /api/analyze  │    │ • Interactive   │
│ • Status View   │    │ • Health Check  │    │ • Q&A           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Orchestrator   │
                    │                 │
                    │ • Pipeline Mgmt │
                    │ • State Sharing │
                    └─────────────────┘
                             │
                    ┌─────────────────┐
                    │     Agents      │
                    │                 │
                    │ • Ingestion     │
                    │ • Metadata      │
                    │ • Comparison    │
                    │ • Risk Analysis │
                    │ • Report Gen    │
                    └─────────────────┘
                             │
                    ┌─────────────────┐
                    │   MongoDB       │
                    │                 │
                    │ • Review Store  │
                    │ • Analytics     │
                    └─────────────────┘
```

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.11+
- **Frontend**: Streamlit
- **AI/ML**: Groq API, Sentence Transformers, FAISS
- **PDF Processing**: PyMuPDF, PDFPlumber
- **Database**: MongoDB
- **Containerization**: Docker
- **Reporting**: ReportLab

## 📋 Prerequisites

- Python 3.11 or higher
- MongoDB (local or cloud instance)
- Groq API key
- Docker (optional, for containerized deployment)

## 🚀 Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd LexGuard
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   MONGODB_URI=mongodb://localhost:27017/lexguard
   ```

5. **Start MongoDB** (if running locally)
   ```bash
   mongod
   ```

### Docker Deployment

1. **Build the container**
   ```bash
   docker build -t lexguard .
   ```

2. **Run with Docker**
   ```bash
   docker run -p 8000:8000 -p 8501:8501 -e GROQ_API_KEY=your_key lexguard
   ```

## 🎯 Usage

### Web Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app_agent.py
   ```
   Access at: http://localhost:8501

2. **Upload a PDF contract** and view real-time analysis progress

3. **Download the generated report** or export JSON data

### API Usage

1. **Start the FastAPI server**
   ```bash
   uvicorn main:app --reload
   ```
   API docs at: http://localhost:8000/docs

2. **Analyze a contract**
   ```bash
   curl -X POST "http://localhost:8000/api/analyze" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@contract.pdf"
   ```

### Chatbot Interface

```bash
python chatbot_agent.py
```

## 📁 Project Structure

```
LexGuard/
├── main.py                 # FastAPI backend
├── app_agent.py           # Streamlit web app
├── chatbot_agent.py       # Chatbot interface
├── orchestrator.py        # Pipeline orchestrator
├── agent_state.py         # Shared state management
├── database.py            # MongoDB integration
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── .env                  # Environment variables
└── agents/               # AI agent modules
    ├── document_ingestion_agent.py
    ├── metadata_extraction_agent.py
    ├── clause_comparison_agent.py
    ├── risk_classification_agent.py
    └── report_generation_agent.py
```

## 🤖 Agent Pipeline Details

### 1. Document Ingestion Agent
- Extracts text from PDF pages using PyMuPDF
- Detects contract type through keyword matching
- Segments clauses by identifying headings and structure

### 2. Metadata Extraction Agent
- Identifies contract parties, effective dates, and amounts
- Uses AI for entity recognition and classification

### 3. Clause Comparison Agent
- Compares extracted clauses against standard templates
- Flags deviations and provides detailed explanations

### 4. Risk Classification Agent
- Analyzes clauses for legal risks and compliance issues
- Assigns severity levels and generates risk register

### 5. Report Generation Agent
- Compiles all findings into professional PDF reports
- Includes executive summaries, risk assessments, and recommendations

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq AI API key | Required |
| `MONGODB_URI` | MongoDB connection string | `mongodb://localhost:27017/lexguard` |
| `MAX_FILE_SIZE_MB` | Maximum PDF file size | `20` |

### Contract Types Supported

- NDA (Non-Disclosure Agreement)
- SLA (Service Level Agreement)
- MSA (Master Service Agreement)
- And more through extensible keyword signatures

## 🧪 Testing

```bash
# Run the pipeline with a test PDF
python orchestrator.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Groq](https://groq.com/) for fast AI inference
- PDF processing powered by [PyMuPDF](https://pymupdf.readthedocs.io/)
- UI framework: [Streamlit](https://streamlit.io/)

## 📞 Support

For questions or support, please open an issue on GitHub or contact the development team.

---

**LexGuard** - Making contract review intelligent, automated, and reliable.</content>
<parameter name="filePath">c:\Users\VH0000805\Downloads\Hackathon\LexGuard\README.md