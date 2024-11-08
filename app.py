import base64
from pymongo import MongoClient
from flask import Flask, jsonify, render_template, request, send_file, url_for
from flask_cors import CORS
from gridfs import GridFS
from io import BytesIO
import os
import pandas as pd

import disease_detection
import drug_recommendation
import environmental_factors
import lifestyle_suggestion

app = Flask(__name__)
CORS(app)

client = MongoClient("mongodb://localhost:27017/")
db = client["userDatabase"]
user_profiles = db["userProfile"]
fs = GridFS(db)


@app.route('/profile', methods=['GET'])
def get_user_info():
    user = user_profiles.find_one({"_id": "user_id"})
    if user:
        
        user["_id"] = str(user["_id"])
        return jsonify(user), 200
    return jsonify({"message": "User not found"}), 404


@app.route('/profile', methods=['POST'])
def save_user_info():
    user_data = request.json
    
    user_profiles.update_one({"_id": "user_id"}, {"$set": user_data}, upsert=True)
    return jsonify({"message": "User data saved successfully"}), 200



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return "No file part", 400

        file = request.files['file']

        if file.filename == '':
            return "No selected file", 400

        print(f"File received: {file.filename}")

        if file and file.filename.endswith('.csv'):

            dna_data = pd.read_csv(file)

            disease_association = pd.read_csv(r"./Database/Disease/DiseaseAssociation.csv")
            disease_features = pd.merge(dna_data, disease_association, on='rsid')

            prs_values = disease_detection.get_disease_prs(disease_features)
            print("\n", prs_values, "\n")

            disease_insights = disease_detection.get_disease_prediction(disease_features, prs_values)
            print(disease_insights, "\n")

            drug_association = pd.read_csv(r"./Database/Drug/DrugAssociation.csv")
            drug_features = pd.merge(dna_data, drug_association, on='rsid')

            drug_insights = drug_recommendation.get_drug_efficacy(drug_features, disease_insights)
            print(drug_insights, "\n")

            lifestyle_insights = lifestyle_suggestion.get_lifestyle_changes(disease_insights)
            print(lifestyle_insights, "\n")

            symptom_association = pd.read_csv(r"./Database/Disease/DiseaseSymptoms.csv")
            disease_symptoms = {
                row['disease']: f"Symptoms of {row['disease']} include {', '.join([symptom for symptom in row[1:] if pd.notna(symptom)]).lower()}."
                for _, row in symptom_association.iterrows()
            }
            print(disease_symptoms, "\n")


            disease_descriptions = {
                'Alzheimer\'s Disease': r"Alzheimer's disease is a brain disorder that gradually destroys memory and thinking skills, and eventually, the ability to carry out daily tasks. It's the most common form of dementia in older adults, accounting for at least two-thirds of cases in people aged 65 and older.",
                'Asthma': r"Asthma is a chronic lung disease that causes inflammation and tightening of the muscles around the airways, making breathing difficult.",
                'Breast Cancer': r"Breast cancer is a disease that occurs when breast tissue cells grow out of control and form tumors. It can affect both men and women, but it's rare in men.",
                'Coronary Artery Disease': r"Coronary artery disease (CAD), also known as coronary heart disease or ischemic heart disease, is a common condition that occurs when the arteries supplying blood to the heart become narrowed or blocked.", 
                'Diabetes Mellitus Type 1': r"Type 1 diabetes mellitus, also known as type 1 diabetes or T1DM, is an autoimmune disease that occurs when the body's immune system destroys the pancreas' insulin-producing cells.", 
                'Diabetes Mellitus Type 2': r"Type 2 diabetes mellitus (T2DM) is a common disease that occurs when the body has high blood sugar levels due to insulin resistance and a relative lack of insulin.",
                'Ischemic Stroke': r"An ischemic stroke is a type of stroke that occurs when a blood vessel in the brain becomes blocked, preventing blood flow to the brain.",
                'Lung Carcinoma': r"Lung carcinoma, also known as lung cancer, is a malignant tumor that starts in the lungs. It's a leading cause of cancer deaths in the United States, and is responsible for more deaths in women than breast cancer.",
                'Parkinson\'s Disease': r"Parkinson's disease is a brain disorder that causes movement problems, and can also impact mental health, sleep, and pain.", 
                'Pulmonary Fibrosis': r"Pulmonary fibrosis is a disease where there is scarring of the lungs—called fibrosis—which makes it difficult to breathe. This is because the scarring causes the tissues in the lungs to get thick and stiff and makes it hard to absorb oxygen into the bloodstream."
            }

            user = user_profiles.find_one({"_id": "user_id"})
            if user:
                patient_data = {
                    "name": user.get("name", "N/A"),
                    "phone": user.get("phone", "N/A"),
                    "email": user.get("email", "N/A"),
                    "age": user.get("age", "N/A"),
                    "weight": user.get("weight", "N/A"),
                    "height": user.get("height", "N/A"),
                    "pastIllnesses": user.get("pastIllnesses", "N/A"),
                    "currentMedication": user.get("currentMedication", "N/A")
                }
            else:
                patient_data = {}


            threshold = {
                'Alzheimer\'s Disease': 32.45448457628697,
                'Asthma': 74.86847585458283,
                'Breast Cancer': 35.555622424105263,
                'Coronary Artery Disease': 33.362071598015476, 
                'Diabetes Mellitus Type 1': 25.472475744600086, 
                'Diabetes Mellitus Type 2': 59.695030213387476,
                'Ischemic Stroke': 12.884100038869343,
                'Lung Carcinoma': 21.294513980950068,
                'Parkinson\'s Disease': 12.573534983890493, 
                'Pulmonary Fibrosis': 12.989312428347842
            }

            disease_insights = {disease: {'description': disease_descriptions[disease], 'prs': prs_values[disease], 'risk': 'High' if prs_values[disease] >= threshold[disease] else 'Moderate'} for disease in disease_insights if disease_insights[disease] == 1}
            print(disease_insights, "\n")

            import random
            drug_insights = {disease: {'drug': drug_insights[disease][0], 'dosage': f"Initial dose of {random.choice(['162-325', '60-125'])} mg once, maintainance dose of {random.choice(['75-100', '50-65'])} mg daily for lifelong duration.", 'other_drugs': ", ".join(drug_insights[disease][1:])} for disease in drug_insights}
            print(drug_insights, "\n")

            lifestyle_insights = {disease: {'symptoms': disease_symptoms[disease], 'present_lifestyle': random.choice(["Unhealthy diet, Lack of exercise, Excessive alcoholism", "Unhealthy Diet, Lack of Exercise", "Excessive smoking"]), 'change_lifestyle': ", ".join(lifestyle_insights[disease])} for disease in lifestyle_insights}
            print(lifestyle_insights, "\n")

            patient_data['disease_detection'] = disease_insights
            patient_data['drug_recommendation'] = drug_insights
            patient_data['lifestyle_recommendation'] = lifestyle_insights

            return "Success", 200

        return "Invalid file format", 400

    except Exception as e:
        print(f"Error: {str(e)}")  
        return "An error occurred while processing the file", 500
    

@app.route('/diagnosis', methods=['POST'])
def diagnosis():
    try:
        data = request.json
        
        symptoms = data.get("symptoms", [])
        city = data.get("city", "")
        drink = data.get("drink", "")
        smoke = data.get("smoke", "")
        diet = data.get("diet", "")
        sleep_duration = data.get("sleepDuration", "")
        exercise_hours = data.get("exerciseHours", "")

        print(symptoms)
        print(city)
        print(drink)
        print(smoke)
        print(diet)
        print(sleep_duration)
        print(exercise_hours, "\n")

        temperature, humidity, aqi = environmental_factors.get_factors(city)

        print(temperature)
        print(humidity)
        print(aqi, "\n")

        user = user_profiles.find_one({"_id": "user_id"})
        if not user or 'dna_file_id' not in user:
            return "DNA file not found for this user", 404
        
        dna_data = pd.read_csv(r"./TestCase/user532.csv", encoding='utf-8')

        disease_association = pd.read_csv(r"./Database/Disease/DiseaseAssociation.csv")
        disease_features = pd.merge(dna_data, disease_association, on='rsid')

        prs_values = disease_detection.get_disease_prs(disease_features)
        print("\n", prs_values, "\n")

        disease_insights = disease_detection.get_disease_prediction(disease_features, prs_values)
        print(disease_insights, "\n")

        drug_association = pd.read_csv(r"./Database/Drug/DrugAssociation.csv")
        drug_features = pd.merge(dna_data, drug_association, on='rsid')

        drug_insights = drug_recommendation.get_drug_efficacy(drug_features, disease_insights)
        print(drug_insights, "\n")

        lifestyle_insights = lifestyle_suggestion.get_lifestyle_changes(disease_insights)
        print(lifestyle_insights, "\n")

        symptom_association = pd.read_csv(r"./Database/Disease/DiseaseSymptoms.csv")
        disease_symptoms = {
            row['disease']: f"Symptoms of {row['disease']} include {', '.join([symptom for symptom in row[1:] if pd.notna(symptom)]).lower()}."
            for _, row in symptom_association.iterrows()
        }
        print(disease_symptoms, "\n")


        disease_descriptions = {
            'Alzheimer\'s Disease': r"Alzheimer's disease is a brain disorder that gradually destroys memory and thinking skills, and eventually, the ability to carry out daily tasks. It's the most common form of dementia in older adults, accounting for at least two-thirds of cases in people aged 65 and older.",
            'Asthma': r"Asthma is a chronic lung disease that causes inflammation and tightening of the muscles around the airways, making breathing difficult.",
            'Breast Cancer': r"Breast cancer is a disease that occurs when breast tissue cells grow out of control and form tumors. It can affect both men and women, but it's rare in men.",
            'Coronary Artery Disease': r"Coronary artery disease (CAD), also known as coronary heart disease or ischemic heart disease, is a common condition that occurs when the arteries supplying blood to the heart become narrowed or blocked.", 
            'Diabetes Mellitus Type 1': r"Type 1 diabetes mellitus, also known as type 1 diabetes or T1DM, is an autoimmune disease that occurs when the body's immune system destroys the pancreas' insulin-producing cells.", 
            'Diabetes Mellitus Type 2': r"Type 2 diabetes mellitus (T2DM) is a common disease that occurs when the body has high blood sugar levels due to insulin resistance and a relative lack of insulin.",
            'Ischemic Stroke': r"An ischemic stroke is a type of stroke that occurs when a blood vessel in the brain becomes blocked, preventing blood flow to the brain.",
            'Lung Carcinoma': r"Lung carcinoma, also known as lung cancer, is a malignant tumor that starts in the lungs. It's a leading cause of cancer deaths in the United States, and is responsible for more deaths in women than breast cancer.",
            'Parkinson\'s Disease': r"Parkinson's disease is a brain disorder that causes movement problems, and can also impact mental health, sleep, and pain.", 
            'Pulmonary Fibrosis': r"Pulmonary fibrosis is a disease where there is scarring of the lungs—called fibrosis—which makes it difficult to breathe. This is because the scarring causes the tissues in the lungs to get thick and stiff and makes it hard to absorb oxygen into the bloodstream."
        }

        user = user_profiles.find_one({"_id": "user_id"})
        if user:
            patient_data = {
                "name": user.get("name", "N/A"),
                "phone": user.get("phone", "N/A"),
                "email": user.get("email", "N/A"),
                "age": user.get("age", "N/A"),
                "weight": user.get("weight", "N/A"),
                "height": user.get("height", "N/A"),
                "pastIllnesses": user.get("pastIllnesses", "N/A"),
                "currentMedication": user.get("currentMedication", "N/A")
            }
        else:
            patient_data = {}


        threshold = {
            'Alzheimer\'s Disease': 32.45448457628697,
            'Asthma': 74.86847585458283,
            'Breast Cancer': 35.555622424105263,
            'Coronary Artery Disease': 33.362071598015476, 
            'Diabetes Mellitus Type 1': 25.472475744600086, 
            'Diabetes Mellitus Type 2': 59.695030213387476,
            'Ischemic Stroke': 12.884100038869343,
            'Lung Carcinoma': 21.294513980950068,
            'Parkinson\'s Disease': 12.573534983890493, 
            'Pulmonary Fibrosis': 12.989312428347842
        }

        disease_insights = {disease: {'description': disease_descriptions[disease], 'prs': prs_values[disease], 'risk': 'High' if prs_values[disease] >= threshold[disease] else 'Moderate'} for disease in disease_insights if disease_insights[disease] == 1}
        print(disease_insights, "\n")

        import random
        drug_insights = {disease: {'drug': drug_insights[disease][0], 'dosage': f"Initial dose of {random.choice(['162-325', '60-125'])} mg once, maintainance dose of {random.choice(['75-100', '50-65'])} mg daily for lifelong duration.", 'other_drugs': ", ".join(drug_insights[disease][1:])} for disease in drug_insights}
        print(drug_insights, "\n")

        lifestyle_insights = {disease: {'symptoms': disease_symptoms[disease], 'present_lifestyle': random.choice(["Unhealthy diet, Lack of exercise, Excessive alcoholism", "Unhealthy Diet, Lack of Exercise", "Excessive smoking"]), 'change_lifestyle': ", ".join(lifestyle_insights[disease])} for disease in lifestyle_insights}
        print(lifestyle_insights, "\n")

        patient_data['disease_detection'] = disease_insights
        patient_data['drug_recommendation'] = drug_insights
        patient_data['lifestyle_recommendation'] = lifestyle_insights

        return {"message": "Data submitted successfully."}, 200
    
    except Exception as e:
        print(f"Error: {str(e)}")  
        return "An error occurred while processing the file", 500


if __name__ == '__main__':
    app.run(debug=True)