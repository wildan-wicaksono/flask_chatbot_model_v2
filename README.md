# 🤖 Mental Health Chatbot with Machine Learning (Bahasa Indonesia)

Repositori ini berisi implementasi chatbot berbasis Machine Learning yang mampu memahami dan merespons pertanyaan pengguna dengan empati menggunakan **FastText bahasa Indonesia** untuk representasi kata. Chatbot ini cocok untuk layanan mental health, chatbot konsultatif, atau aplikasi interaktif lain berbasis teks dalam bahasa Indonesia.

## 🎯 Tujuan Proyek

- Memberikan dukungan emosional berbasis chatbot dalam Bahasa Indonesia
- Mengklasifikasikan intent pengguna menggunakan NLP
- Merespons dengan kalimat yang empatik dan sesuai konteks

---

## 🛠️ Teknologi yang Digunakan

- Python 3.x
- TensorFlow & Keras
- **FastText Bahasa Indonesia** (pretrained atau custom embedding)
- Flask (untuk API)
- Pickle (serialisasi model dan objek)
- JSON (untuk dataset intents)
- Heroku (opsional, deployment)

## 📚 Tentang FastText

FastText digunakan untuk representasi kata dalam bentuk vektor embedding. Model ini membantu mengenali konteks kata dalam Bahasa Indonesia, meskipun terdapat typo atau variasi bentuk kata (subword embeddings).

**Sumber model**: [FastText pre-trained for Indonesian (cc.id.300.vec)](https://fasttext.cc/docs/en/crawl-vectors.html)

---

## 📁 Struktur Repositori

| File/Folder            | Fungsi                                                |
| ---------------------- | ----------------------------------------------------- |
| `app.py`               | Script utama menjalankan Flask API                    |
| `chatbot_model.h5`     | Model hasil training (menggunakan FastText embedding) |
| `tokenizer.pickle`     | Tokenizer teks                                        |
| `label_encoder.pickle` | Encoder label untuk klasifikasi intent                |
| `intents.json`         | Dataset intent dan ekspresi pengguna                  |
| `responses.pkl`        | Daftar respons berdasarkan intent                     |
| `requirements.txt`     | Dependensi proyek                                     |
| `Procfile`             | Deployment Heroku                                     |

---

## 🔁 Langkah Replikasi

### 1. Clone Repositori

```bash
git clone https://github.com/nuansarahardian/nama-repo.git
cd nama-repo
```

### 2. Setup Virtual Environment (opsional)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Catatan:** Pastikan file model FastText (`cc.id.300.vec` atau `.bin`) telah disiapkan dan dimuat dalam kode `app.py`.

## 🔍 Contoh API

### Endpoint: `/predict` (POST)

```json
{
  "message": "Aku stres banget karena tugas kuliah menumpuk"
}
```

**Response:**

```json
{
  "intent": "stress_due_to_academic",
  "response": "Aku bisa ngerti kok rasa tertekan karena tugas. Mau aku bantu kasih tips manajemen stres?"
}
```

---

## ⚙️ Model Training (Opsional)

1. Tokenisasi data dari `intents.json`
2. Gunakan embedding FastText saat membangun model di TensorFlow
3. Simpan model sebagai `chatbot_model.h5`
4. Simpan tokenizer dan label encoder (`.pickle`)

---

## Deployment ke Railway

Langkah-langkah:

1. Masuk ke https://railway.app dan login

2. Klik New Project > Deploy from GitHub Repo

3. Pilih repositori flask_chatbot_model_v2

4. Railway akan otomatis mendeteksi app.py dan requirements.txt

5. Di tab Settings, pastikan:

6. Port: 5000

7. Start Command: python app.py

## 👥 Kontributor

- Nuansa Rahardian
- Wildan Bagus Wicaksono
- Nelson Lau
