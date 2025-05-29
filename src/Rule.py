import numpy as np

class AsthmaPredictor:
    def __init__(self, model):
        self.model = model

    def rule_engine(self, age, bmi, smoking, wheezing, nighttime, sob, chest, coughing, exercise):
        if smoking == 1 and sob == 1 and coughing == 1:
            return 0.95
        if wheezing == 1 and nighttime == 1 and chest == 1:
            return 0.92
        if exercise == 1 and sob == 1:
            return 0.88
        if bmi >= 30 and sob == 1:
            return 0.87
        if age >= 60 and (sob == 1 or chest == 1):
            return 0.89
        if 18 <= age <= 40 and bmi < 28 and smoking == 1 and sob == 1:
            return 0.83
        if coughing == 1 and wheezing == 1:
            return 0.81
        if nighttime == 1 and coughing == 1:
            return 0.79
        if smoking == 1 and wheezing == 1 and chest == 1:
            return 0.85
        if 10 <= age <= 17 and exercise == 1 and coughing == 1:
            return 0.80
        if bmi < 25 and smoking == 0 and sob == 0 and coughing == 0 and wheezing == 0:
            return 0.10
        return 0.20
    
    def generate_advice(self, age, bmi, smoking, wheezing, nighttime, sob, chest, coughing, exercise):
        advice = []
        if smoking == 1:
            advice.append("Consider reducing or quitting smoking to improve respiratory health.")
        if bmi >= 30:
            advice.append("Maintaining a healthy BMI may reduce asthma-related risks.")
        if sob == 1:
            advice.append("Shortness of breath indicates possible airway issues—consider consulting a pulmonologist.")
        if exercise == 1 and sob == 1:
            advice.append("Avoid intense workouts until symptoms are under control.")
        if nighttime == 1 or coughing == 1:
            advice.append("Monitor nighttime symptoms and avoid allergens before sleep.")
        if wheezing == 1:
            advice.append("Wheezing may indicate airway inflammation—an inhaler might be helpful.")
        if chest == 1:
            advice.append("Persistent chest tightness should not be ignored—consider a medical checkup.")
        if not advice:
            advice.append("You're showing no major risk factors. Keep maintaining a healthy lifestyle!")
        return advice

    def predict(self, age, bmi, smoking, wheezing, nighttime, sob, chest, coughing, exercise):
        rule_prob = self.rule_engine(age, bmi, smoking, wheezing, nighttime, sob, chest, coughing, exercise)
        features = np.array([[age, bmi, smoking, wheezing, nighttime, sob, chest, coughing, exercise]])
        xgb_prob = self.model.predict_proba(features)[0][1]
        final_prob = round(0.6 * xgb_prob + 0.4 * rule_prob, 2)
        label = "High risk of Asthma" if final_prob >= 0.5 else "Low risk of Asthma"
        emoji = "✅" if label.startswith("Low") else "⚠️"
        advice_list = self.generate_advice(age, bmi, smoking, wheezing, nighttime, sob, chest, coughing, exercise)

        advice_text = "\n".join(advice_list)
        return f"{emoji} {label}. (Confidence: {final_prob:.2f})\n\nAdvice:\n{advice_text}"
