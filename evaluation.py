import google.generativeai as genai
import json

genai.configure(api_key='AIzaSyCA_bPi_VBL4pZQjVuXIUDoT-HpI5KccyQ')

def evaluate_diagnosis(student_diagnosis, patient_info):
    """
    Evaluate the student's diagnosis using the Gemini API.
    
    :param student_diagnosis: str, the diagnosis provided by the student
    :param patient_info: dict, containing patient information
    :return: dict, containing evaluation results and feedback
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = f"""You are an expert medical professional tasked with evaluating a medical student's interaction with a virtual patient.

    Patient Information:
    Medical Specialty: {patient_info['medical_specialty']}
    Sample Name: {patient_info['sample_name']}
    Description: {patient_info['description']}
    Keywords: {patient_info['keywords']}

    Student's Diagnosis:
    {student_diagnosis}

    Please evaluate the student's performance based on the following metrics and their respective weightage:

    1. Diagnostic Accuracy (35% of total score):
       - Correct identification of primary health issue
       - Appropriate differential diagnoses
       - Recognition of key symptoms and their significance

    2. History Taking Skills (15% of total score):
       - Thoroughness of questioning
       - Relevance of questions to the presenting problem
       - Ability to elicit important details

    3. Communication Skills (5% of total score):
       - Clarity of questions and explanations
       - Empathy and rapport building
       - Active listening and appropriate follow-up questions

    4. Clinical Reasoning (5% of total score):
       - Logical progression of questioning
       - Appropriate prioritization of health issues
       - Consideration of psychosocial factors

    5. Suggested Medications and Tests (35% of total score):
       - Appropriateness and accuracy of suggested medications
       - Relevance and necessity of suggested tests

    6. Professionalism (5% of total score):
       - Respectful interaction
       - Maintaining appropriate boundaries
       - Time management

    Calculate the total score out of 100 based on the percentages given.

    Format your response as a JSON object with keys: "score", "explanation", "suggestions", and "correct_diagnosis". """

    response = model.generate_content(prompt)
    
    try:
        # Safely parse the response as JSON
        evaluation = json.loads(response.text)
        
        # Ensure all required keys are present
        required_keys = ["score", "explanation", "suggestions", "correct_diagnosis"]
        if all(key in evaluation for key in required_keys):
            return evaluation
        else:
            # If not all keys are present, return a default response
            return {
                "score": 0,
                "explanation": "Error in evaluation. Please try again.",
                "suggestions": "N/A",
                "correct_diagnosis": "Unable to determine"
            }
    except json.JSONDecodeError:
        # If the response is not in valid JSON format, return a default response
        return {
            "score": 0,
            "explanation": "Error in evaluation. Please try again.",
            "suggestions": "N/A",
            "correct_diagnosis": "Unable to determine"
        }
    except Exception as e:
        # Catch any other exceptions and return a default response
        print(f"An error occurred: {e}")
        return {
            "score": 0,
            "explanation": "Error in evaluation. Please try again.",
            "suggestions": "N/A",
            "correct_diagnosis": "Unable to determine"
        }

def provide_feedback(evaluation):
    """
    Format the feedback based on the evaluation results.
    
    :param evaluation: dict, the evaluation results from evaluate_diagnosis
    :return: str, formatted feedback message
    """
    feedback = f"""
    Evaluation of your diagnosis:
    
    Score: {evaluation.get('score', 0)}/100
    
    Explanation: {evaluation.get('explanation', 'Error in evaluation. Please try again.')}
    
    Suggestions for improvement: {evaluation.get('suggestions', 'N/A')}
    
    Correct diagnosis: {evaluation.get('correct_diagnosis', 'Unable to determine')}
    """
    return feedback
