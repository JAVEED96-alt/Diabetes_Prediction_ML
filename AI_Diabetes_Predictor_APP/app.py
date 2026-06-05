from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load ML model + scaler
def load_model():
    try:
        model_path = 'model.pkl'
        scaler_path = 'scaler.pkl'
          
        # Check if files exist
        if not os.path.exists(model_path):
            print(f"ERROR: {model_path} not found!")
            return None, None
            
        if not os.path.exists(scaler_path):
            print(f"ERROR: {scaler_path} not found!")
            return None, None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            print("✓ Model loaded successfully")
            
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            print("✓ Scaler loaded successfully")
            
        return model, scaler
    except Exception as e:
        print(f"ERROR loading model/scaler: {e}")
        return None, None

model, scaler = load_model()


def generate_suggestions(input_data, risk_level):
    suggestions = []
    
    glucose = input_data['glucose']
    bmi = input_data['bmi']
    age = input_data['age']
    bp = input_data['blood_pressure']
    insulin = input_data['insulin']
    skin = input_data['skin_thickness']
    fam = input_data['diabetes_pedigree']

    # Risk Level Overview
    suggestions.append("━━━━━━━━━━━━━━━━━")
    suggestions.append("📊 RISK ASSESSMENT SUMMARY")
    suggestions.append("━━━━━━━━━━━━━━━━━")
    
    if risk_level == "Low":
        suggestions.append("✅ GOOD NEWS: Your diabetes risk is LOW")
        suggestions.append("Continue your healthy lifestyle to maintain this low risk.")
    elif risk_level == "Moderate":
        suggestions.append("⚠️ ATTENTION: Your diabetes risk is MODERATE")
        suggestions.append("Take preventive action now to avoid progression to diabetes.")
    else:
        suggestions.append("🚨 URGENT: Your diabetes risk is HIGH")
        suggestions.append("Immediate medical consultation is strongly recommended.")
    
    suggestions.append("")
    suggestions.append("━━━━━━━━━━━━━━━━")
    suggestions.append("🩺 PERSONALIZED HEALTH RECOMMENDATIONS")
    suggestions.append("━━━━━━━━━━━━━━━━")
    suggestions.append("")

    # Glucose Analysis
    suggestions.append("🔬 GLUCOSE LEVEL ANALYSIS:")
    if glucose >= 126:
        suggestions.append("   ❗ Your fasting glucose (≥126 mg/dL) indicates DIABETES")
        suggestions.append("   → Schedule HbA1c test immediately with your doctor")
        suggestions.append("   → Start monitoring blood sugar daily")
        suggestions.append("   → Eliminate all sugary foods and refined carbs")
    elif glucose >= 100:
        suggestions.append("   ⚠️ Your glucose (100-125 mg/dL) indicates PREDIABETES")
        suggestions.append("   → Get HbA1c test within 1-2 weeks")
        suggestions.append("   → Cut out sugary drinks, sweets, and white bread")
        suggestions.append("   → Start 30-minute walks after meals")
    else:
        suggestions.append("   ✓ Your glucose level is in the normal range")
        suggestions.append("   → Maintain balanced diet to keep it stable")
    suggestions.append("")

    # BMI Analysis
    suggestions.append("⚖️ BODY MASS INDEX (BMI) ANALYSIS:")
    if bmi >= 30:
        suggestions.append(f"   ❗ Your BMI ({bmi:.1f}) indicates OBESITY")
        suggestions.append("   → Target: Lose 5-10% body weight in 6 months")
        suggestions.append("   → Diet: Reduce portions by 25%, increase vegetables")
        suggestions.append("   → Exercise: Start with 20-min walks, build up to 45 mins daily")
    elif 25 <= bmi < 30:
        suggestions.append(f"   ⚠️ Your BMI ({bmi:.1f}) indicates OVERWEIGHT")
        suggestions.append("   → Losing just 3-5 kg can reduce diabetes risk by 50%")
        suggestions.append("   → Focus on portion control and regular exercise")
    elif 18.5 <= bmi < 25:
        suggestions.append(f"   ✓ Your BMI ({bmi:.1f}) is in the healthy range")
        suggestions.append("   → Maintain current weight through balanced diet")
    else:
        suggestions.append(f"   ⚠️ Your BMI ({bmi:.1f}) is low - consult a nutritionist")
    suggestions.append("")

    # Blood Pressure Analysis
    suggestions.append("💓 BLOOD PRESSURE ANALYSIS:")
    if bp >= 90:
        suggestions.append(f"   ❗ Your BP ({bp} mm Hg) is HIGH")
        suggestions.append("   → Reduce salt intake to <5g per day")
        suggestions.append("   → Avoid processed/packaged foods")
        suggestions.append("   → Check BP weekly and consult doctor if consistently high")
    elif 80 <= bp < 90:
        suggestions.append(f"   ⚠️ Your BP ({bp} mm Hg) is ELEVATED")
        suggestions.append("   → Limit salt, avoid fried foods")
        suggestions.append("   → Practice stress management (yoga, meditation)")
    else:
        suggestions.append(f"   ✓ Your BP ({bp} mm Hg) is in normal range")
    suggestions.append("")

    # Insulin Analysis
    if insulin > 166:
        suggestions.append("💉 INSULIN LEVEL ANALYSIS:")
        suggestions.append(f"   ⚠️ Your insulin level ({insulin} µU/mL) is HIGH")
        suggestions.append("   → May indicate insulin resistance")
        suggestions.append("   → Consult endocrinologist for detailed evaluation")
        suggestions.append("")

    # Skin Thickness (Insulin Resistance Indicator)
    if skin > 30:
        suggestions.append("📏 SKIN THICKNESS INDICATOR:")
        suggestions.append(f"   ⚠️ Elevated skin thickness ({skin} mm) may indicate insulin resistance")
        suggestions.append("   → Discuss with your doctor for proper assessment")
        suggestions.append("")

    # Family History
    if fam >= 1.5:
        suggestions.append("👨‍👩‍👧‍👦 FAMILY HISTORY ALERT:")
        suggestions.append("   ⚠️ Strong family history of diabetes detected")
        suggestions.append("   → Get screened annually (HbA1c + fasting glucose)")
        suggestions.append("   → Lifestyle modification is CRITICAL for prevention")
        suggestions.append("")

    # Age Factor
    if age >= 60:
        suggestions.append("📅 AGE-RELATED RECOMMENDATIONS:")
        suggestions.append(f"   ⚠️ At age {age}, diabetes risk increases significantly")
        suggestions.append("   → Get comprehensive diabetes screening every 6 months")
        suggestions.append("   → Focus on gentle exercise like walking, swimming")
        suggestions.append("")
    elif age >= 45:
        suggestions.append("📅 AGE-RELATED RECOMMENDATIONS:")
        suggestions.append(f"   At age {age}, regular screening is important")
        suggestions.append("   → Annual diabetes screening recommended")
        suggestions.append("")

    # Lifestyle Recommendations
    suggestions.append("━━━━━━━━━━━━━━━━━━━━━━")
    suggestions.append("🥗 ESSENTIAL LIFESTYLE CHANGES")
    suggestions.append("━━━━━━━━━━━━━━━━━━━━━━━")
    suggestions.append("")
    
    suggestions.append("DIET MODIFICATIONS:")
    suggestions.append("   ✓ Eat whole grains instead of white rice/bread")
    suggestions.append("   ✓ Include vegetables in every meal (fill half your plate)")
    suggestions.append("   ✓ Choose lean proteins: fish, chicken, legumes")
    suggestions.append("   ✓ Avoid: Sugary drinks, sweets, fried foods, processed snacks")
    suggestions.append("   ✓ Drink 8-10 glasses of water daily")
    suggestions.append("")
    
    suggestions.append("EXERCISE ROUTINE:")
    suggestions.append("   ✓ Walk 30-45 minutes daily (best after meals)")
    suggestions.append("   ✓ Do light strength training 2-3 times per week")
    suggestions.append("   ✓ Take stairs instead of elevators")
    suggestions.append("   ✓ Stand up and move every hour if desk job")
    suggestions.append("")
    
    suggestions.append("MONITORING & CHECK-UPS:")
    if risk_level == "High":
        suggestions.append("   ✓ Visit doctor within 1 week")
        suggestions.append("   ✓ Get HbA1c test immediately")
        suggestions.append("   ✓ Monitor blood glucose daily")
        suggestions.append("   ✓ Follow up every 3 months")
    elif risk_level == "Moderate":
        suggestions.append("   ✓ Schedule doctor appointment within 2 weeks")
        suggestions.append("   ✓ Get HbA1c and lipid profile tests")
        suggestions.append("   ✓ Check blood glucose monthly")
        suggestions.append("   ✓ Follow up every 6 months")
    else:
        suggestions.append("   ✓ Annual health check-up recommended")
        suggestions.append("   ✓ Maintain healthy lifestyle habits")
    suggestions.append("")

    # Call to Action
    suggestions.append("━━━━━━━━━━━━━━━")
    suggestions.append("🎯 NEXT STEPS")
    suggestions.append("━━━━━━━━━━━━━━━")
    if risk_level == "High":
        suggestions.append("1. 👨‍⚕️ Book doctor appointment TODAY")
        suggestions.append("2. 🎥 Watch educational videos below")
        suggestions.append("3. 📋 Start implementing diet changes immediately")
        suggestions.append("4. 📱 Download a glucose tracking app")
    elif risk_level == "Moderate":
        suggestions.append("1. 👨‍⚕️ Schedule doctor visit this week")
        suggestions.append("2. 🎥 Learn prevention strategies from videos")
        suggestions.append("3. 🏃 Start daily 30-minute walks")
        suggestions.append("4. 📊 Track your progress monthly")
    else:
        suggestions.append("1. 🎥 Stay informed through health videos")
        suggestions.append("2. 💪 Maintain active lifestyle")
        suggestions.append("3. 🥗 Continue healthy eating habits")
    suggestions.append("")
    
    suggestions.append("⚠️ IMPORTANT DISCLAIMER:")
    suggestions.append("This assessment is for educational purposes only and does NOT")
    suggestions.append("replace professional medical diagnosis. Always consult a")
    suggestions.append("qualified healthcare provider for medical advice and treatment.")

    return "\n".join(suggestions)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/assessment", methods=["GET", "POST"])
def assessment():
    prediction = None
    error_message = None
    suggestions_text = None
    risk_level = None

    if request.method == "POST":
        # Check if model and scaler are loaded
        if model is None or scaler is None:
            error_message = "Model or scaler files are missing! Please ensure model.pkl and scaler.pkl are in the project directory."
            return render_template("assessment.html", error_message=error_message)
        
        try:
            input_data = {
                'pregnancies': 0,  # Automatically set to 0
                'glucose': float(request.form.get("glucose", 0)),
                'blood_pressure': float(request.form.get("blood_pressure", 0)),
                'skin_thickness': float(request.form.get("skin_thickness", 0)),
                'insulin': float(request.form.get("insulin", 0)),
                'bmi': float(request.form.get("bmi", 0)),
                'diabetes_pedigree': float(request.form.get("diabetes_pedigree", 0)),
                'age': float(request.form.get("age", 0))
            }

            # Validate input data
            if any(v < 0 for v in input_data.values()):
                error_message = "All values must be positive numbers."
                return render_template("assessment.html", error_message=error_message)

            features = np.array([[input_data[key] for key in input_data]])
            features_scaled = scaler.transform(features)

            proba = float(model.predict_proba(features_scaled)[0][1])
            
            if proba < 0.20:
                risk_level = "Low"
                emoji = "✅"
            elif proba < 0.50:
                risk_level = "Moderate"
                emoji = "⚠️"
            else:
                risk_level = "High"
                emoji = "🚨"

            prediction = f"{emoji} {risk_level} Risk of Diabetes (Probability: {proba:.1%})"
            suggestions_text = generate_suggestions(input_data, risk_level)

        except ValueError as e:
            error_message = f"Invalid input values. Please enter numeric values only. Error: {e}"
        except Exception as e:
            error_message = f"An error occurred during prediction: {str(e)}"

    return render_template("assessment.html", prediction=prediction, error_message=error_message, suggestions=suggestions_text, risk_level=risk_level)


@app.route("/comprehensive", methods=["GET", "POST"])
def comprehensive():
    prediction = None
    error_message = None
    suggestions_text = None
    risk_level = None

    if request.method == "POST":
        # Check if model and scaler are loaded
        if model is None or scaler is None:
            error_message = "Model or scaler files are missing! Please ensure model.pkl and scaler.pkl are in the project directory."
            return render_template("comprehensive.html", error_message=error_message)
        
        try:
            input_data = {
                'pregnancies': float(request.form.get("pregnancies", 0)),
                'glucose': float(request.form.get("glucose", 0)),
                'blood_pressure': float(request.form.get("blood_pressure", 0)),
                'skin_thickness': float(request.form.get("skin_thickness", 0)),
                'insulin': float(request.form.get("insulin", 0)),
                'bmi': float(request.form.get("bmi", 0)),
                'diabetes_pedigree': float(request.form.get("diabetes_pedigree", 0)),
                'age': float(request.form.get("age", 0))
            }

            features = np.array([[input_data[key] for key in input_data]])
            features_scaled = scaler.transform(features)

            proba = float(model.predict_proba(features_scaled)[0][1])
            
            if proba < 0.20:
                risk_level = "Low"
                emoji = "✅"
            elif proba < 0.50:
                risk_level = "Moderate"
                emoji = "⚠️"
            else:
                risk_level = "High"
                emoji = "🚨"

            prediction = f"{emoji} {risk_level} Risk of Diabetes (Probability: {proba:.1%})"
            suggestions_text = generate_suggestions(input_data, risk_level)

        except Exception as e:
            error_message = f"Error: {e}"

    return render_template("comprehensive.html", prediction=prediction, error_message=error_message, suggestions=suggestions_text, risk_level=risk_level)


@app.route("/videos")
def videos():
    search_links = [
        ("What to eat with diabetes", "https://www.youtube.com/results?search_query=what+to+eat+for+diabetes"),
        ("Exercise for diabetes control", "https://www.youtube.com/results?search_query=diabetes+exercise"),
        ("How to reverse diabetes naturally", "https://www.youtube.com/results?search_query=reverse+diabetes+naturally"),
        ("Diabetes diet plan", "https://www.youtube.com/results?search_query=diabetes+diet+plan"),
        ("Understanding blood sugar levels", "https://www.youtube.com/results?search_query=blood+sugar+levels+explained"),
        ("Diabetes medication guide", "https://www.youtube.com/results?search_query=diabetes+medication+guide")
    ]
    return render_template("videos.html", search_links=search_links)


@app.route("/doctors")
def doctors():
    doctors_list = [
        {
            "name": "Dr. Anish Behl",
            "specialization": "General Physician — Diabetes Management",
            "experience": "28+ years",
            "rating": "4.8★",
            "clinic": "Apollo BGS Hospitals",
            "area": "Kuvempunagar, Mysuru",
            "link": "https://www.practo.com/mysore/doctor/anish-behl-diabetologist"
        },
        {
            "name": "Dr. Guru Prasad B V",
            "specialization": "Diabetes & Hypertension",
            "experience": "20+ years",
            "rating": "4.7★",
            "clinic": "Apollo Clinic",
            "area": "Vani Vilas Mohalla, Mysuru",
            "link": "https://www.practo.com/mysore/doctor/guru-prasad-b-v-general-physician"
        },
        {
            "name": "Dr. Rajesh Kumar",
            "specialization": "Endocrinologist - Diabetes Specialist",
            "experience": "15+ years",
            "rating": "4.6★",
            "clinic": "Columbia Asia Hospital",
            "area": "Mysuru",
            "link": "https://www.practo.com/mysore"
        }
    ]
    return render_template("doctors.html", doctors=doctors_list)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)