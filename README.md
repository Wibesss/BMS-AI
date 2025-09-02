# BMS-AI – AI Video Summarizer

BMS-AI je aplikacija koja omogućava **automatsku transkripciju, diarizaciju govornika i sažimanje razgovora iz video snimaka**. Kombinuje modele za **prepoznavanje govora (Whisper)**, **detekciju govornika (KMeans + MFCC)** i **sažimanje teksta (HuggingFace Transformers)**, dok frontend (React + Vite) omogućava jednostavno korišćenje preko web interfejsa.

---

## Funkcionalnosti
- Ekstrakcija zvuka iz video fajla
- Automatska diarizacija (prepoznavanje govornika)
- Transkripcija govora pomoću OpenAI Whisper modela
- Generisanje sažetka pomoću Flan-T5 ili BART-SAMSum modela
- REST API (Flask) za interakciju sa AI servisom
- Web interfejs (React + Vite) za upload i prikaz rezultata

---

## Struktura projekta
```
BMS-AI/
│── backend/
│   ├── createTranscript.py   # Transkripcija + diarizacija
│   ├── summarizer.py         # Summarization logika
│   ├── app.py                # Flask server (API rute)
│   ├── requirements.txt      # Python zavisnosti
│
│── frontend/                 # React + Vite aplikacija
│   ├── src/
│   ├── package.json
│
│── uploads/                  # Uploadovani fajlovi
│── README.md
```

---

## Instalacija i pokretanje

### Backend (Flask + ML modeli)
```bash
git clone https://github.com/Wibesss/BMS-AI.git
cd BMS-AI/backend
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python app.py
```
Server ce raditi na `http://localhost:5000`

### Frontend (React + Vite)
```bash
cd ../frontend
npm install
npm run dev
```
Frontend ce raditi na `http://localhost:5173`

---

## API rute
### 1. Upload fajla
```
POST /upload
```
**Body (form-data):**
- `file`: video fajl (.mp4, .avi, …)

### 2. Sažimanje videa
```
POST /summarize
```
**Body (form-data):**
- `file_path`: putanja fajla (iz /upload)
- `num_of_speakers`: broj govornika

**Odgovor (JSON):**
```json
{
  "summarized": "Summary of the conversation..."
}
```

---

## Korišćene tehnologije
- **Backend:** Python, Flask, Whisper, Librosa, Scikit-learn, HuggingFace Transformers, Torch
- **Frontend:** React, Vite, TailwindCSS
- **AI modeli:** Whisper, Flan-T5, BART SAMSum

---

## Plan za budućnost
- [ ] Automatsko prepoznavanje broja govornika
- [ ] Poboljšana diarizacija (pyannote.audio)
- [ ] Podrška za više jezika
- [ ] Optimizacija modela za real-time rad
- [ ] Deployment (Docker, AWS/GCP)

---

