import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

file_path = 'dataset.csv'
data = pd.read_csv(file_path)

X = data.drop(columns=['diabetes', 'case_number'])
y = data['diabetes']

categorical_cols = ['gender', 'smoking_history']
numerical_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, k_neighbors=1)),  # Adjust k_neighbors based on minority class size
    ('classifier', RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Function to send email
def send_email(to_email, subject, body):
    from_email = 'no.reply.etms@gmail.com' 
    from_password = 'npns zdah yvoh vftw'  

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, from_password)
    text = msg.as_string()
    server.sendmail(from_email, to_email, text)
    server.quit()

def main():
    case_number = input("Enter the Case Number: ")
    age = input("Enter the Age: ")
    email = input("Please Enter the Email: ")
    phone = input("Please Confirm the phone number: ")

    case_data = data[data['case_number'] == int(case_number)]

    if case_data.empty:
        print("Case Number not found.")
        return

    case_data = case_data.drop(columns=['diabetes', 'case_number'])

    prediction = model.predict(case_data)
    diabetes = prediction[0]

    if diabetes == 1:
        recommendations = ("The patient with Case Number {} may be affected with diabetes. "
                           "Please consult a doctor immediately and follow the below precautions:\n"
                           "1. Follow a healthy diet that is suitable for diabetics.\n"
                           "2. Use supplemental insulin, if necessary.\n"
                           "3. Regular exercise is highly recommended.\n"
                           "4. Regularly monitor blood glucose levels.\n"
                           "5. Avoid sugary and high-carbohydrate foods.").format(case_number)
    else:
        recommendations = ("The patient with Case Number {} does not appear to be affected with diabetes based on the model prediction. "
                           "However, it is always good to maintain a healthy lifestyle. Here are some general recommendations:\n"
                           "1. Maintain a balanced diet.\n"
                           "2. Exercise regularly.\n"
                           "3. Monitor your health periodically with routine check-ups.").format(case_number)

    # Send email
    subject = "Health Report for Case Number {}".format(case_number)
    body = "Dear User,\n\n{}\n\nBest regards,\nHealth Monitoring System".format(recommendations)
    send_email(email, subject, body)
    print("Email sent successfully!")

if __name__ == "__main__":
    main()
