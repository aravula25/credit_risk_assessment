from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('loan_default_model.pkl')

def validate_input(field, type_fn=int, default=0):
    """ Helper function to safely convert form input to specified type or return a default """
    try:
        return type_fn(request.form.get(field, ''))
    except ValueError:
        return default

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Gather data from the form using the validation helper
            person_age = validate_input('person_age')
            person_income = validate_input('person_income')
            person_emp_length = validate_input('person_emp_length', float, 0.0)
            loan_amnt = validate_input('loan_amnt')
            loan_int_rate = validate_input('loan_int_rate', float, 0.0)
            loan_percent_income = validate_input('loan_percent_income', float, 0.0)
            cb_person_cred_hist_length = validate_input('cb_person_cred_hist_length')

            # Extract strings safely
            person_home_ownership = request.form.get('person_home_ownership', '').upper()
            loan_intent = request.form.get('loan_intent', '').replace(" ", "").upper()
            loan_grade = request.form.get('loan_grade', '').upper()
            cb_person_default_on_file = request.form.get('cb_person_default_on_file', '').upper()

            input_data = pd.DataFrame({
                'person_age': [person_age],
                'person_income': [person_income],
                'person_emp_length': [person_emp_length],
                'loan_amnt': [loan_amnt],
                'loan_int_rate': [loan_int_rate],
                'loan_percent_income': [loan_percent_income],
                'cb_person_cred_hist_length': [cb_person_cred_hist_length],
                f'person_home_ownership_{person_home_ownership}': [1],
                f'loan_intent_{loan_intent}': [1],
                f'loan_grade_{loan_grade}': [1],
                f'cb_person_default_on_file_{cb_person_default_on_file}': [1]
            })

            expected_columns = [
    'person_age', 'person_income', 'person_emp_length', 'loan_amnt',
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
    'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER',
    'person_home_ownership_OWN', 'person_home_ownership_RENT',
    'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION',
    'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
    'loan_intent_PERSONAL', 'loan_intent_VENTURE', 'loan_grade_A',
    'loan_grade_B', 'loan_grade_C', 'loan_grade_D', 'loan_grade_E',
    'loan_grade_F', 'loan_grade_G', 'cb_person_default_on_file_N',
    'cb_person_default_on_file_Y'
]


            # Ensure all expected columns are present
            for column in expected_columns:
                if column not in input_data.columns:
                    input_data[column] = 0

            input_data = input_data[expected_columns]

            prediction = model.predict(input_data)
            return render_template('index.html', prediction=prediction[0])
        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
