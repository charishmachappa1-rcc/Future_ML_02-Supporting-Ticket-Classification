import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = {
    "ticket": [
        "Unable to login to my account",
        "Payment failed during checkout",
        "Website is loading very slowly",
        "Need refund for my order",
        "Password reset not working",
        "App crashes on startup"
    ],
    "category": [
        "Login",
        "Payment",
        "Performance",
        "Refund",
        "Login",
        "Technical"
    ]
}

df = pd.DataFrame(data)

print(df)
X = df["ticket"]
y = df["category"]
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_vector = vectorizer.fit_transform(X)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_vector = vectorizer.fit_transform(X)
print(X_vector.toarray())
X_train, X_test, y_train, y_test = train_test_split(
    X_vector,
    y,
    test_size=0.2,
    random_state=42
)
print("Training Data:", len(y_train))
print("Testing Data:", len(y_test))
model = MultinomialNB()

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

print(classification_report(y_test, predictions,zero_division=0))
def get_priority(ticket):
    ticket = ticket.lower()

    if "crash" in ticket or "failed" in ticket or "urgent" in ticket:
        return "High"
    elif "slow" in ticket or "refund" in ticket:
        return "Medium"
    else:
        return "Low"

df["Priority"] = df["ticket"].apply(get_priority)

print(df)
print("\nSupport Ticket Classification Completed Successfully!")

import matplotlib.pyplot as plt

# Count the number of tickets in each category
category_counts = df["category"].value_counts()

# Create bar chart
plt.bar(category_counts.index, category_counts.values)

# Add title and labels
plt.title("Support Ticket Categories")
plt.xlabel("Category")
plt.ylabel("Number of Tickets")

# Display the chart
plt.show()

