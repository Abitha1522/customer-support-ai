import os
import warnings
import pandas as pd
from transformers import pipeline

warnings.filterwarnings("ignore")

print("=" * 80)
print("SMART REVIEW ANALYZER & CUSTOMER CARE AI")
print("=" * 80)

# --------------------------------------------------
# 1. DATA INGESTION
# --------------------------------------------------

print("\n[1] Loading customer support dataset...")

# Automatically use the current project folder
project_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(project_dir, "customer_support_tickets.csv")

if not os.path.exists(dataset_path):
    print("ERROR: customer_support_tickets.csv not found.")
    print(f"Expected location: {dataset_path}")
    raise SystemExit

df = pd.read_csv(dataset_path)

print(f"Dataset loaded successfully!")
print(f"Total records: {len(df)}")

# --------------------------------------------------
# 2. DATA PREPROCESSING
# --------------------------------------------------

print("\n[2] Preprocessing data...")

# Remove records without required text/customer information
df = df.dropna(subset=["Ticket Description", "Customer Name"]).copy()

# Clean ticket descriptions
df["cleaned_text"] = (
    df["Ticket Description"]
    .astype(str)
    .str.lower()
    .str.strip()
)

print(f"Records after preprocessing: {len(df)}")

# --------------------------------------------------
# 3. MACHINE LEARNING LAYER
# --------------------------------------------------

print("\n[3] Classifying tickets into support departments...")


def classify_department(text):
    text = str(text).lower()

    technical_words = [
        "crash", "bug", "software", "error",
        "unusable", "freeze", "code",
        "technical", "login", "system"
    ]

    billing_words = [
        "price", "subscription", "expensive",
        "cost", "billing", "charge",
        "refund", "pay", "payment"
    ]

    logistics_words = [
        "late", "shipping", "delivery",
        "arrived", "packaging", "lost",
        "shipment", "damaged", "order"
    ]

    if any(word in text for word in technical_words):
        return "Technical Support"

    elif any(word in text for word in billing_words):
        return "Billing & Sales"

    elif any(word in text for word in logistics_words):
        return "Logistics & Operations"

    else:
        return "General Support Desk"


df["assigned_department"] = df["cleaned_text"].apply(
    classify_department
)

print("Ticket department classification completed.")

# --------------------------------------------------
# 4. DEEP LEARNING - SENTIMENT ANALYSIS
# --------------------------------------------------

print("\n[4] Loading DistilBERT sentiment model...")
print("The first run may take some time because the model is downloaded.")

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print("Sentiment model loaded successfully.")


def analyze_sentiment(text):
    text = str(text)[:1500]

    result = sentiment_model(text)[0]

    return result["label"], round(result["score"], 4)


print("\nAnalyzing customer sentiment...")

df[["sentiment", "confidence_score"]] = df[
    "cleaned_text"
].apply(
    lambda text: pd.Series(analyze_sentiment(text))
)

print("Sentiment analysis completed.")

# --------------------------------------------------
# 5. AI RESPONSE GENERATION
# --------------------------------------------------

print("\n[5] Generating AI-based customer responses...")


def generate_response(row):

    customer = row["Customer Name"]
    department = row["assigned_department"]
    sentiment = row["sentiment"]

    if sentiment == "NEGATIVE":

        if department == "Technical Support":
            return (
                f"Hi {customer}, we are sorry for the technical "
                "issue you experienced. Our technical support team "
                "will review the problem and assist you as soon as possible."
            )

        elif department == "Billing & Sales":
            return (
                f"Hi {customer}, we understand your concern regarding "
                "the billing issue. Our billing team will review the "
                "details and assist you with the resolution."
            )

        elif department == "Logistics & Operations":
            return (
                f"Hi {customer}, we apologize for the delivery or "
                "order-related issue. Our operations team will review "
                "the shipment details and assist you."
            )

        else:
            return (
                f"Hi {customer}, we are sorry that your experience "
                "did not meet expectations. Our support team will "
                "review your concern and assist you."
            )

    else:
        return (
            f"Hi {customer}, thank you for contacting our support team. "
            "We appreciate your feedback and are happy to assist you."
        )


df["ai_draft_response"] = df.apply(
    generate_response,
    axis=1
)

print("AI response generation completed.")

# --------------------------------------------------
# 6. DISPLAY SAMPLE RESULTS
# --------------------------------------------------

print("\n" + "=" * 80)
print("CUSTOMER SUPPORT AI - SAMPLE RESULTS")
print("=" * 80)

# Display first 10 tickets in terminal
for index, row in df.head(10).iterrows():

    print(f"\nTicket ID       : {row['Ticket ID']}")
    print(f"Customer        : {row['Customer Name']}")
    print(f"Ticket Type     : {row['Ticket Type']}")
    print(f"Description     : {row['Ticket Description'][:120]}...")
    print(f"Department      : {row['assigned_department']}")
    print(f"Sentiment       : {row['sentiment']}")
    print(f"Confidence      : {row['confidence_score'] * 100:.2f}%")
    print(f"AI Response     : {row['ai_draft_response']}")
    print("-" * 80)

# --------------------------------------------------
# 7. SAVE OUTPUT
# --------------------------------------------------

print("\n[6] Saving final analysis...")

output_path = os.path.join(
    project_dir,
    "smart_review_analyzer_output.xlsx"
)

df.to_excel(
    output_path,
    index=False
)

print("\n" + "=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 80)

print(f"\nOutput file created:")
print(output_path)

print(f"\nTotal tickets processed: {len(df)}")
print("Excel analysis file generated successfully.")