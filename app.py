from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Sample training data
emails = [
    "Congratulations! You won a free prize",
    "Claim your free money now",
    "You have won a lottery",
    "Click this link to get your reward",
    "Win cash by clicking this link",
    "Exclusive offer! Get free products now",
    "Meeting is scheduled for tomorrow",
    "Please submit your assignment",
    "Can you send me the project report?",
    "Your appointment is confirmed",
    "Let's meet at 5 PM",
    "Please find the document attached"
]

labels = [
    "spam",
    "spam",
    "spam",
    "spam",
    "spam",
    "spam",
    "not spam",
    "not spam",
    "not spam",
    "not spam",
    "not spam",
    "not spam"
]

# Convert emails into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train the model
model = MultinomialNB()
model.fit(X, labels)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        email = request.form["email"]

        email_vector = vectorizer.transform([email])
        prediction = model.predict(email_vector)[0]

        if prediction == "spam":
            prediction = "🚨 This email is SPAM!"
        else:
            prediction = "✅ This email is NOT SPAM!"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)