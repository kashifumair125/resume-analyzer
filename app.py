from flask import Flask, request, render_template, jsonify
import os
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Predefined keywords for different job roles
JOB_KEYWORDS = {
    'java developer': {
        'essential': ['java', 'spring', 'hibernate', 'junit', 'maven', 'gradle'],
        'optional': ['microservices', 'rest', 'docker', 'kubernetes', 'aws', 'jpa']
    },
    'data scientist': {
        'essential': ['python', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'deep learning'],
        'optional': ['sql', 'r', 'matplotlib', 'seaborn', 'keras', 'pytorch']
    },
    'frontend developer': {
        'essential': ['html', 'css', 'javascript', 'react', 'angular', 'vue.js'],
        'optional': ['typescript', 'sass', 'webpack', 'redux', 'graphql', 'd3.js']
    },
    'backend developer': {
        'essential': ['node.js', 'express', 'sql', 'mongodb', 'restful apis', 'docker'],
        'optional': ['redis', 'aws', 'kubernetes', 'nginx', 'java', 'python']
    },
    'full stack developer': {
        'essential': ['html', 'css', 'javascript', 'react', 'node.js', 'express', 'sql'],
        'optional': ['typescript', 'docker', 'aws', 'graphql', 'webpack', 'kubernetes']
    },
    'devops engineer': {
        'essential': ['linux', 'docker', 'kubernetes', 'aws', 'ci/cd', 'terraform'],
        'optional': ['ansible', 'git', 'nginx', 'prometheus', 'grafana', 'jenkins']
    }
}

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_text_from_pdf(file_path):
    return extract_text(file_path)

def get_job_keywords(job_title):
    keywords = JOB_KEYWORDS.get(job_title.lower(), {})
    return keywords.get('essential', []), keywords.get('optional', [])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    try:
        if 'resume' not in request.files or 'job_description' not in request.form or 'job_title' not in request.form:
            return jsonify({'error': 'Resume file, job description, or job title is missing'}), 400

        resume_file = request.files['resume']
        job_description = request.form['job_description']
        job_title = request.form['job_title']
        
        if resume_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if resume_file and resume_file.filename.endswith('.pdf'):
            filename = 'resume.pdf'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            resume_file.save(file_path)
            
            resume_text = extract_text_from_pdf(file_path)
            
            # Create TF-IDF Vectorizer and compute similarity
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            similarity_score = similarity_matrix[0][0] * 100

            # Identify missing and optional keywords
            essential_keywords, optional_keywords = get_job_keywords(job_title)
            
            missing_keywords = [kw for kw in essential_keywords if kw.lower() not in resume_text.lower()]
            optional_keywords_found = [kw for kw in optional_keywords if kw.lower() in resume_text.lower()]

            suggestions = ''
            if missing_keywords:
                suggestions += f"<div class='missing-keywords'>Missing Keywords: {', '.join(missing_keywords)}</div>"
            if optional_keywords_found:
                suggestions += f"<div class='optional-keywords'>Optional Keywords Found: {', '.join(optional_keywords_found)}</div>"

            return jsonify({'similarity_score': round(similarity_score, 2), 'suggestions': suggestions})

        return jsonify({'error': 'Invalid file format'}), 400

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=False,host='0.0.0.0')
