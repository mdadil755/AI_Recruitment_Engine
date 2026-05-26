from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import pandas as pd
from groq import Groq
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
load_dotenv()

# 1. Initialize Flask App
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
db_pass = os.getenv("DB_PASSWORD")
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://postgres:{db_pass}@localhost:5432/recruitment_db'

db = SQLAlchemy(app)

class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    role = db.Column(db.String(100))
    experience = db.Column(db.Integer)
    result = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(300), nullable=False)

# 2. Groq AI Setup
client_ai = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 3. Load Model
try:
    model = pickle.load(open('job_model_v8.pkl', 'rb'))
except FileNotFoundError:
    print("Error: job_model_v8.pkl file is not found.")

# 4. Role-Skills Database
role_skills_map = {
    "Data Scientist":  ["python", "machine learning", "statistics", "pandas", "scikit-learn", "deep learning"],
    "Data Analyst":    ["sql", "excel", "powerbi", "python", "statistics", "tableau"],
    "ML Engineer":     ["python", "tensorflow", "mlops", "pytorch", "aws", "docker"],
    "Full Stack Dev":  ["react", "nodejs", "mongodb", "express", "javascript", "git"],
    "Backend Dev":     ["python", "django", "postgresql", "flask", "redis", "docker"],
    "Frontend Dev":    ["javascript", "react", "tailwind", "html", "css", "nextjs"],
    "App Developer":   ["flutter", "dart", "firebase", "kotlin", "swift", "ui/ux"],
    "DevOps Engineer": ["docker", "kubernetes", "linux", "jenkins", "terraform", "aws"],
    "Cloud Architect": ["aws", "networking", "iam", "azure", "gcp", "terraform"],
    "Data Engineer":   ["spark", "sql", "etl", "hadoop", "kafka", "airflow"],
    "Cyber Security":  ["networking", "linux", "ethical hacking", "wireshark", "metasploit", "siem"],
    "Ethical Hacker":  ["web security", "python", "linux", "burpsuite", "nmap", "owasp"],
    "AI Researcher":   ["pytorch", "calculus", "research papers", "python", "cuda", "nlp"],
    "SOC Analyst":     ["incident response", "splunk", "log analysis", "siem", "networking", "firewalls"],
    "IT Support":      ["hardware", "troubleshooting", "os", "active directory", "office 365", "linux"]
}

# 5. Domains
domains = ["None (Fresher)"] + list(role_skills_map.keys()) + ["Sales", "Marketing", "HR", "Finance"]


# 6. AI Recommendation Function
def get_ai_recommendation(name, role, status, missing_skills, exp_years):
    try:
        prompt = f"""
        You are an expert HR AI assistant. Give a short 5-6 line recommendation:
        Candidate: {name}
        Applied For: {role}
        Result: {status}
        Experience: {exp_years} years
        Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}

        If Rejected: Motivate and suggest how to learn missing skills.
        If Selected: Congratulate and suggest what to focus on next.
        Do NOT mention any alternative scenarios.
        IMPORTANT: Use HTML <b> tags to bold the missing skills. Never use asterisks (**).
        Keep it professional, crisp and encouraging.
        """
        response = client_ai.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


# --- ROUTES ---

@app.route('/')
def login_page():
    return render_template('login.html')


@app.route('/login_action', methods=['POST'])
def login_action():
    email = request.form.get('email')
    password = request.form.get('password')

    user = User.query.filter_by(email=email).first()

    if user and check_password_hash(user.password, password):
        session['logged_in'] = True
        session['user_name'] = user.name
        return redirect(url_for('prediction_page'))
    else:
        return render_template('login.html', error="Invalid Email or Password")

@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/signup_action', methods=['POST'])
def signup_action():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')

    existing = User.query.filter_by(email=email).first()

    if existing:
        return render_template('signup.html', error="Email already registered")

    hashed_password = generate_password_hash(password)

    new_user = User(
        name=name,
        email=email,
        password=hashed_password
    )

    db.session.add(new_user)
    db.session.commit()

    return redirect(url_for('login_page'))

@app.route('/prediction')
def prediction_page():
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))

    result         = session.pop('result', None)
    confidence     = session.pop('confidence', None)
    name           = session.pop('name', None)
    missing        = session.pop('missing', None)
    recommendation = session.pop('recommendation', None)

    return render_template('index.html',
                           roles=role_skills_map.keys(),
                           domains=domains,
                           roles_dict=role_skills_map,
                           result=result,
                           confidence=confidence,
                           name=name,
                           missing=missing,
                           recommendation=recommendation)


@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))

    name          = request.form.get('name')
    applied_role  = request.form.get('role')
    past_domain   = request.form.get('past_domain') or 'None (Fresher)'
    exp_years     = int(request.form.get('exp_years', 0))
    qualification = request.form.get('qualification')
    user_skills   = request.form.getlist('skills')
    internship    = 1 if request.form.get('internship') else 0
    projects      = 1 if request.form.get('projects') else 0

    skills_str = ", ".join(user_skills) if user_skills else "none"

    input_data = pd.DataFrame(
        [[applied_role, past_domain, exp_years, qualification, skills_str, internship, projects]],
        columns=['applied_role', 'past_domain', 'exp_years', 'qualification', 'skills', 'internship', 'projects']
    )

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    confidence  = round(float(max(probability)) * 100, 2)

    required_skills = set(role_skills_map.get(applied_role, []))
    attained_skills = set(user_skills)
    missing_skills  = list(required_skills - attained_skills)

    status = "Selected 🎉" if prediction == 1 else "Rejected ❌"

    record = Candidate(
    name=name,
    role=applied_role,
    experience=exp_years,
    result=status,
    confidence=confidence
    )

    db.session.add(record)
    db.session.commit()

    recommendation = get_ai_recommendation(name, applied_role, status, missing_skills, exp_years)

    session['result']         = status
    session['confidence']     = confidence
    session['name']           = name
    session['missing']        = missing_skills
    session['recommendation'] = recommendation

    return redirect(url_for('prediction_page'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)