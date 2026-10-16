# 🎓 StudyMate AI — Your Personalized AI Learning Assistant

**Learn it. Practice it. Get evaluated. Improve it.**

StudyMate AI is an AI-powered learning assistant designed to make studying, revision, and interview preparation easier and more interactive. It allows learners to work with their study materials, ask questions, generate practice quizzes and flashcards, prepare for interviews, and explore images with AI assistance—all through a single application.

Built using Python, Streamlit, Retrieval-Augmented Generation (RAG), vector search, and Large Language Models (LLMs), StudyMate AI helps learners turn their study materials into an interactive learning experience.

---

## 📌 Project Overview

Students often spend a significant amount of time reading lengthy documents, preparing revision notes, creating practice questions, and preparing for technical interviews.

StudyMate AI brings these learning activities together in one application. It uses semantic search and AI-generated responses to help users interact with their learning materials and practise important concepts.

### 🎯 Key Objectives

- Simplify learning from study documents.
- Provide context-aware answers to questions about uploaded materials.
- Automate quiz and flashcard generation for revision.
- Support technical, HR, and project-based interview practice.
- Help users understand images through AI-assisted explanations.
- Make self-paced learning more interactive and accessible.

---

## ✨ Features

### 1. 📚 My Notes — Document Management

- Work with study documents and online PDF sources.
- Extract text from supported documents for further processing.
- Organize document content for semantic search and question answering.
- Maintain document information for subsequent learning activities.

### 2. 💬 Ask My Notes — RAG-Based Question Answering

- Ask questions related to your study materials.
- Retrieve relevant document chunks using semantic similarity search.
- Generate AI responses using retrieved context.
- Use embeddings and a vector database to find relevant information instead of relying only on keyword matching.

**How it works:** Your question is converted into an embedding, relevant document chunks are retrieved from the vector database, and the retrieved context is provided to the language model to generate an answer.

### 3. 📝 AI Quiz Generator

- Generate practice questions from learning materials.
- Customize quiz topics and difficulty levels.
- Choose the number of questions where supported.
- Use quizzes for self-assessment and revision.
- Review quiz history where available.

### 4. 🗂️ AI Flashcards

- Generate flashcards to revise important concepts.
- Support difficulty-based revision.
- Track cards marked as known or needing revision.
- Review flashcard session history.

### 5. 🎤 Interview Practice

- Practise technical, HR, mixed, and project-based interviews.
- Answer questions in an interactive text-based workflow.
- Submit answers for AI-assisted evaluation.
- Receive scores and feedback to help identify areas for improvement.
- Review previous practice sessions where available.

### 6. 🖼️ AI Image Explanation

- Upload an image for AI-assisted analysis.
- Ask questions about the image.
- Receive explanations based on the capabilities of the configured AI model.
- Access saved image-related interactions where supported.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive web application |
| Retrieval-Augmented Generation (RAG) | Context-aware question answering |
| Sentence Transformers | Text embeddings |
| `all-MiniLM-L6-v2` | Semantic embedding model |
| ChromaDB | Persistent vector storage and similarity search |
| Large Language Models (LLMs) | Answer generation and AI-assisted learning features |
| PyPDF | PDF text extraction |
| Git and GitHub | Version control and project hosting |

The exact LLM provider depends on the application's configuration. API credentials should be stored securely in environment variables rather than committed to GitHub.

---

## 🧠 How StudyMate AI Works

```text
        Study Documents / Online PDFs
                    |
                    v
          Text Extraction
                    |
                    v
         Text Chunking
                    |
                    v
       Sentence Transformer
          Embeddings
                    |
                    v
            ChromaDB
         Vector Database
                    |
                    v
         User Asks a Question
                    |
                    v
        Semantic Similarity
              Search
                    |
                    v
        Retrieve Relevant
           Document Chunks
                    |
                    v
       LLM Generates Response
                    |
                    v
          Context-Aware Answer
```

The same learning-material foundation supports features such as question answering, quiz generation, and flashcard creation.

---

## 🏗️ Project Structure

```text
StudyMate-AI/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── src/
│   ├── document_loader.py
│   ├── rag_pipeline.py
│   ├── rag_qa.py
│   ├── quiz_generator.py
│   ├── flashcard_generator.py
│   ├── interview_practice.py
│   └── ...
│
└── screenshots/
    ├── Home.png
    ├── AskMyNotes.png
    ├── MyNotes.png
    ├── Quizzes.png
    ├── FlashCards.png
    ├── InterviewPractice.png
    └── ImageExplanation.png
```

*Note: This is a representative structure. Keep the filenames and folders in this section aligned with the files actually uploaded to the repository.*

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version supported by the dependencies.
- pip package manager.
- Git, if cloning the repository.
- API credentials for the configured LLM provider, if required.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/studymate-ai.git
cd studymate-ai
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv studymate_env
studymate_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file locally if the application uses environment variables for its API credentials.

For example:

```env
GEMINI_API_KEY=your_api_key_here
```

Use the actual variable name expected by your application. Never commit your real API key or `.env` file to GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

---


---

## 🔐 Security and Privacy

- API keys and credentials must remain outside the public repository.
- The `.env` file and virtual environment folders should not be uploaded.
- Generated vector databases and private learning documents should be excluded unless there is a specific, safe reason to share them.
- Users should review AI-generated responses for accuracy.

---

## 🔮 Future Improvements

Potential enhancements include:

- More robust source citations and document-grounded answers.
- Improved personalization based on learning progress.
- Topic-wise performance analytics and weak-area identification.
- Expanded document format support.
- Enhanced voice-based interview practice and speech-to-text evaluation.
- Deployment for easy access through a public web interface.

These are future enhancement ideas, not claims that every feature is already implemented.

---

## 🎯 Learning Outcomes

This project demonstrates practical application of:

- Python application development.
- Document processing and text chunking.
- Natural Language Processing (NLP) concepts.
- Semantic embeddings and vector databases.
- Retrieval-Augmented Generation (RAG).
- LLM integration.
- Interactive application development using Streamlit.
- Building AI-assisted educational workflows.

---

## 👩‍💻 Author

**Yogita Patil**

MCA | Aspiring Data Scientist / Machine Learning Engineer

Interested in Data Science, Machine Learning, Deep Learning, NLP, Generative AI, and building practical AI applications.

---

## ⭐ Support

If you find this project interesting, consider starring the repository on GitHub.

**StudyMate AI — Learn smarter, practise better, and prepare with confidence.**
