import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import datetime

# Load spaCy's English model for NER
nlp = spacy.load("en_core_web_sm")

# Expanded sample data: posts with corresponding labels and timestamps
posts_data = [
    {"text": "Completed C++ certification offered by XYZ Institute", "timestamp": "2021-01-15"},
    {"text": "Learned Data Structures and Algorithms from ABC University", "timestamp": "2021-06-20"},
    {"text": "Built web apps using JavaScript and React", "timestamp": "2021-09-10"},
    {"text": "Completed Machine Learning certification at Coursera", "timestamp": "2022-03-05"},
    {"text": "Started Python programming course provided by Udemy", "timestamp": "2022-07-22"},
    {"text": "Built an Android app for restaurant management", "timestamp": "2022-11-15"},
    {"text": "Completed Web Development bootcamp at CodeAcademy", "timestamp": "2023-02-18"},
    {"text": "Enrolled in Data Science course offered by DataCamp", "timestamp": "2023-06-01"},
    {"text": "Completed Cybersecurity certification at Cybrary", "timestamp": "2023-09-12"}
]

# Extract texts and labels for training
posts = [post['text'] for post in posts_data]
labels = [
    "certification",
    "course",
    "project",
    "certification",
    "course",
    "project",
    "certification",
    "course",
    "certification"
]

# Convert text to numerical data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(posts)

# Train classifier
clf = SVC(kernel='linear')
clf.fit(X, labels)

# Function to extract information using spaCy
def extract_details(text, category, timestamp):
    doc = nlp(text)
    details = {'category': category, 'timestamp': timestamp}

    if category == "certification":
        # Extract certification topic and company name
        topics = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "GPE", "NORP"]]
        details['certification_topic'] = ' '.join([ent.text for ent in doc.ents if ent.label_ == "PRODUCT"])
        details['certification_company'] = topics

    elif category == "course":
        # Extract course names and providing companies
        course_name = []
        course_provider = []

        # Look for course-related phrases
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"]:
                course_provider.append(ent.text)
            elif ent.label_ == "PERSON" or ent.label_ == "PRODUCT":
                course_name.append(ent.text)

        # Use tokens near "course" keyword if present
        if "course" in text.lower():
            course_index = text.lower().split().index("course")
            course_name = text.split()[course_index - 2:course_index]

        details['course_name'] = ' '.join(course_name)
        details['course_provider'] = course_provider

    elif category == "project":
        # Extract project name and sub-topic
        details['project_name'] = ' '.join(doc.text.split()[1:3])  # Assuming first two words as project name
        details['project_topic'] = [ent.text for ent in doc.ents if ent.label_ == "PRODUCT"]

    return details

# Predict and extract details for each post
extracted_data = []
for post in posts_data:
    text = post['text']
    timestamp = post['timestamp']
    X_new = vectorizer.transform([text])
    predicted_category = clf.predict(X_new)[0]
    extracted_details = extract_details(text, predicted_category, timestamp)
    extracted_data.append(extracted_details)

# Plotting the timeline
timestamps = [datetime.datetime.strptime(data['timestamp'], '%Y-%m-%d') for data in extracted_data]
activities = [data['category'] for data in extracted_data]
labels = [f"{data['category']}: {data['timestamp']}" for data in extracted_data]

plt.figure(figsize=(10, 5))

# Plot each event on the timeline
for i, (timestamp, activity) in enumerate(zip(timestamps, activities)):
    plt.plot(timestamp, i, 'o', markersize=10)
    plt.text(timestamp, i, f' {activity}', verticalalignment='center', fontsize=10)

# Formatting the timeline
plt.yticks(range(len(labels)), labels)
plt.xlabel('Time')
plt.title('Alumni Activity Timeline')
plt.gca().invert_yaxis()  # Optional: invert the y-axis to show the earliest event at the top
plt.grid(axis='x', linestyle='--')
plt.tight_layout()
plt.show()
