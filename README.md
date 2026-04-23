# AI Medical FAQ & Appointment Assistant

A capstone project combining RAG (Retrieval-Augmented Generation) and an AI Agent
to handle medical questions and appointment booking — built with core Python, no frameworks.

## Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| UI          | Streamlit                           |
| API         | FastAPI (Python)                    |
| Agent       | Core Python — intent detect + tools |
| RAG         | sentence-transformers + Pinecone    |
| LLM         | HuggingFace InferenceClient         |
| Database    | PostgreSQL on GCP Cloud SQL         |
| Deployment  | Docker + Kubernetes (GKE)           |
| CI/CD       | GitHub Actions                      |

---

## Project Structure

```
medical_assistant/
├── api/
│   ├── agent.py          # Agent loop, intent detection, slot filling
│   ├── tools.py          # book_appointment, get_appointments, cancel_appointment
│   ├── db.py             # DB connection + schema init + seeding
│   ├── app.py            # FastAPI entry point
│   ├── logger.py         # Structured logging
│   ├── requirements.txt
│   └── Dockerfile
├── rag_service/
│   ├── rag_pipeline.py   # Embed → Pinecone query → LLM answer
│   ├── ingest.py         # One-time document ingestion script
│   ├── app.py            # FastAPI RAG service
│   ├── docs/
│   │   └── medical_faq.txt   # Sample knowledge base
│   ├── requirements.txt
│   └── Dockerfile
├── ui/
│   ├── ui.py             # Streamlit chat interface
│   ├── requirements.txt
│   └── Dockerfile
├── k8s/
│   ├── secrets.yaml
│   ├── api-deployment.yaml
│   ├── rag-deployment.yaml
│   ├── ui-deployment.yaml
│   └── ingest-job.yaml
├── .github/
│   └── workflows/
│       └── deploy.yml
├── .env.example
└── README.md
```

---

## Prerequisites

- Python 3.11+
- Docker Desktop
- kubectl + gcloud CLI
- GCP account (GKE cluster + Cloud SQL Postgres)
- Pinecone account (free tier works)
- HuggingFace account (free API token)

---

## STEP 1 — Set Up External Services

### 1a. Pinecone

1. Go to https://app.pinecone.io and create a free account
2. Create a new Index:
   - Name: `medical-rag`
   - Dimensions: `384`  (matches all-MiniLM-L6-v2)
   - Metric: `cosine`
3. Copy your API key

### 1b. HuggingFace

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is enough)
3. Copy the token

### 1c. GCP Cloud SQL (PostgreSQL)

1. Go to GCP Console → SQL → Create Instance
2. Choose PostgreSQL, name it `medical-db`
3. Set a strong password
4. Under Connections → enable Public IP (for local dev) or Private IP (for GKE)
5. Create a database named `medical_db`
6. Note the Host IP, username (postgres), password, and database name

---

## STEP 2 — Configure Environment

```bash
# Clone or copy this project
cd medical_assistant

# Create your .env file
cp .env.example .env

# Edit .env and fill in all values
nano .env
```

Your `.env` should look like:
```
PINECONE_API_KEY=pc-xxxxxxxx
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxx
DB_HOST=34.xx.xx.xx
DB_NAME=medical_db
DB_USER=postgres
DB_PASSWORD=yourpassword
DB_PORT=5432
RAG_SERVICE_URL=http://localhost:8001
```

---

## STEP 3 — Run Locally (without Docker)

### Terminal 1 — Start RAG service

```bash
cd rag_service
pip install -r requirements.txt

# Copy .env from root
cp ../.env .

# First time only: ingest documents into Pinecone
python ingest.py

# Start RAG service
uvicorn app:app --host 0.0.0.0 --port 8001
```

### Terminal 2 — Start API service

```bash
cd api
pip install -r requirements.txt

# Copy .env from root
cp ../.env .

# This also initialises DB tables and seeds demo data
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Terminal 3 — Start UI

```bash
cd ui
pip install -r requirements.txt

# Set API URL for local dev
export API_URL=http://localhost:8000

streamlit run ui.py
```

Open http://localhost:8501 in your browser.

---

## STEP 4 — Run with Docker Compose (Local Full Stack)

Create `docker-compose.yml` at the root:

```yaml
version: "3.8"
services:
  rag:
    build: ./rag_service
    ports:
      - "8001:8001"
    env_file: .env

  api:
    build: ./api
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      - RAG_SERVICE_URL=http://rag:8001
    depends_on:
      - rag

  ui:
    build: ./ui
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
```

```bash
# Build and start all 3 services
docker-compose up --build

# Open http://localhost:8501
```

---

## STEP 5 — Deploy to GKE

### 5a. Create GKE Cluster

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create cluster
gcloud container clusters create medical-assistant-cluster \
  --zone us-central1-a \
  --num-nodes 2 \
  --machine-type e2-standard-2

# Get credentials
gcloud container clusters get-credentials medical-assistant-cluster \
  --zone us-central1-a
```

### 5b. Build and Push Docker Images

```bash
# Configure Docker for GCR
gcloud auth configure-docker

export PROJECT_ID=your_gcp_project_id

# Build and push all images
docker build -t gcr.io/$PROJECT_ID/medical-api:latest ./api
docker push gcr.io/$PROJECT_ID/medical-api:latest

docker build -t gcr.io/$PROJECT_ID/medical-rag:latest ./rag_service
docker push gcr.io/$PROJECT_ID/medical-rag:latest

docker build -t gcr.io/$PROJECT_ID/medical-ui:latest ./ui
docker push gcr.io/$PROJECT_ID/medical-ui:latest
```

### 5c. Update Image Names in K8s YAMLs

Replace `YOUR_PROJECT_ID` in all k8s/*.yaml files with your actual GCP project ID.

### 5d. Create Kubernetes Secrets

```bash
# Base64-encode each value and fill in k8s/secrets.yaml
echo -n "your_pinecone_key" | base64
echo -n "your_hf_token" | base64
echo -n "your_db_host" | base64
# ... repeat for all values

# Apply secrets
kubectl apply -f k8s/secrets.yaml
```

### 5e. Deploy All Services

```bash
# Deploy in this order
kubectl apply -f k8s/rag-deployment.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/ui-deployment.yaml

# Run the one-time ingest job
kubectl apply -f k8s/ingest-job.yaml

# Watch pods come up
kubectl get pods -w
```

### 5f. Get the Public URL

```bash
kubectl get service ui-service
# Copy the EXTERNAL-IP and open it in your browser on port 80
```

---

## STEP 6 — Set Up CI/CD (GitHub Actions)

1. Push this project to a GitHub repository
2. Go to Settings → Secrets → Actions and add:
   - `GCP_PROJECT_ID` — your GCP project ID
   - `GCP_SA_KEY` — your GCP service account JSON key (base64 encoded)
3. Every push to `main` now automatically builds images and deploys to GKE

---

## Testing the Application

Try these messages in the UI:

**Medical FAQ (RAG):**
- "What are the symptoms of diabetes?"
- "What is hypertension?"
- "What are the side effects of metformin?"

**Booking (Agent + multi-turn):**
- "I have chest pain, I need an appointment"
- "Book me an appointment" (agent will ask for reason)
- "I need to see a doctor for stomach issues tomorrow at 11 AM"

**View / Cancel (Agent):**
- "Show my appointments"
- "Cancel my appointment" (agent will list them with IDs first)
- "Cancel #2"

---

## Architecture Summary

```
User → Streamlit UI → FastAPI API
                          │
               ┌──────────┴──────────┐
               ▼                     ▼
        Intent = faq?         Intent = action?
               │                     │
               ▼                     ▼
        RAG Service            Tool Functions
    (Pinecone + HuggingFace)  (PostgreSQL CRUD)
               │                     │
               └──────────┬──────────┘
                          ▼
                   Response to User
```
