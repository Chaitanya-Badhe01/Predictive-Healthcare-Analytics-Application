# Machine Learning Web Application: Diabetes & Heart Disease Prediction  

**📌 Introduction / Overview**  
This project is a **machine learning-based web application** that predicts the risk of **diabetes and heart disease** using user health data.  
It integrates **Explainable AI (LIME)** to provide transparent and interpretable predictions, ensuring that users and doctors can understand the reasoning behind each prediction.  
The application is built with a **Flask-powered backend** and an **interactive user interface**, making it accessible for both individuals and healthcare professionals.  

**🎯 Problem Statement / Motivation**  
Healthcare risks like diabetes and heart disease are increasing globally.  
Early detection and preventive measures can significantly reduce complications.  
The motivation was to build an AI-powered web app that:  
- Predicts the likelihood of chronic diseases.  
- Provides interpretable predictions for transparency.  
- Supports preventive healthcare decision-making.  

**📂 Data Sources**  
- Publicly available medical datasets on kaggle. 
- User input health data and symptoms.  

**🛠 Tools and Technology**  
- **Languages**: Python  
- **Libraries**:  
  - ML: Scikit-learn, NumPy, Pandas  
  - Explainability: LIME  
  - Visualization: Matplotlib, Seaborn  
- **Frameworks**: Flask (backend), HTML/CSS (frontend)  
- **Platform**: Jupyter Notebook + Flask server  

**🔎 Analysis and Methodology**  
1. **Data Cleaning and Preparation**  
   - Handled missing values and normalized medical parameters.  
   - Encoded categorical variables (gender, smoking history, etc.).  
   - Split data into training and test sets for evaluation.  

2. **Modeling**  
   - Trained multiple models (Random Forest, Neural Networks (CNN)).  
   - Selected **Random Forest Classifier** for high accuracy and robustness.  
   - Saved models as `.pkl` files for deployment in Flask.  

3. **Explainable AI (LIME)**  
   - Integrated LIME to explain individual predictions.  
   - Showed feature contributions (e.g., glucose level, cholesterol) for each prediction.  
   - Improved transparency for users and doctors.  

4. **Web Application Development**  
   - Flask backend handles model predictions and routes.  
   - Interactive frontend form collects user health data.  
   - Results page shows disease risk with LIME explanations.  

**📊 Visualizations**  
- Feature importance plots for disease prediction.  
- LIME plots highlighting top contributing health features.  
- Confusion matrices for model evaluation.  

**🔑 Key Findings & Insights**  
- ✅ Machine learning models can accurately predict risks of diabetes and heart disease.  
- ✅ LIME makes predictions **interpretable**, increasing user trust.  
- ✅ High blood glucose, cholesterol, and hypertension are key indicators.  
- ✅ Random Forest outperformed other models in terms of accuracy.  

**💡 Recommendations**  
- Encourage preventive health check-ups for at-risk patients.  
- Use interpretable AI to guide both patients and doctors in decision-making.  
- Promote healthy lifestyle changes (diet, exercise, reduced smoking/alcohol).  

**📌 Conclusion**  
This project demonstrates the potential of **AI + Explainable AI** in healthcare.  
By combining predictive modeling with transparency, the app bridges the gap between complex algorithms and practical healthcare use.  

**🚀 Future Work**  
- Extend to **multi-disease prediction** (kidney, liver, cancer).  
- Integrate **real-time wearable health data** (Fitbit, Apple Health).  
- Enhance security with **blockchain-based medical records**.  
- Deploy on **cloud platforms** (AWS, GCP, Heroku) for scalability.  

**⚙️ How to Run the Project**  
1. Clone this repository.  
2. Install dependencies:  
   ```bash
   pip install pandas numpy scikit-learn flask lime matplotlib seaborn
