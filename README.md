Smart Review Analyzer & Customer Care AI

An AI-powered customer support ticket analysis system that combines data preprocessing, automated ticket routing, transformer-based sentiment analysis, and AI-generated customer responses.

🚀 Project Overview

This project analyzes customer support tickets and automatically:

Cleans and preprocesses ticket descriptions
Classifies tickets into appropriate support departments
Performs sentiment analysis using DistilBERT
Calculates sentiment confidence scores
Generates customer-specific draft responses
Exports the complete analysis to an Excel file
🧠 Technologies Used
Python
Pandas
PyTorch
Hugging Face Transformers
DistilBERT
OpenPyXL
🔄 Working Process
Customer Support Dataset
          ↓
Data Ingestion
          ↓
Data Preprocessing
          ↓
Ticket Department Classification
          ↓
DistilBERT Sentiment Analysis
          ↓
AI Response Generation
          ↓
Excel Report Generation
📊 Support Categories

The system automatically routes tickets into:

Technical Support
Billing & Sales
Logistics & Operations
General Support Desk
🤖 Sentiment Analysis

The project uses the pretrained:

distilbert-base-uncased-finetuned-sst-2-english

model from Hugging Face to identify whether customer feedback is:

Positive
Negative

The system also records the model's confidence score.

📁 Project Structure
customer-support-ai/
│
├── customer_support_ai.py
├── customer_support_tickets.csv
├── smart_review_analyzer_output.xlsx
├── requirements.txt
└── README.md
⚙️ Installation

Clone the repository:

git clone https://github.com/Abitha1522/customer-support-ai.git
cd customer-support-ai

Install the required Python packages:

pip install -r requirements.txt
▶️ Run the Project

Run:

python customer_support_ai.py

During the first execution, the required DistilBERT model will be downloaded from Hugging Face.

An Excel report will be generated as:

smart_review_analyzer_output.xlsx
📄 Output

The generated analysis contains information including:

Ticket ID
Customer Name
Ticket Type
Ticket Description
Assigned Department
Sentiment
Confidence Score
AI Draft Response
🎯 Project Objective

The objective of this project is to demonstrate how AI and NLP techniques can assist customer support teams by automatically understanding incoming support tickets, routing them to relevant departments, analyzing customer sentiment, and preparing response drafts.

👩‍💻 Author

Abitha K

AI/ML Enthusiast | Python | Machine Learning | NLP | Deep Learning