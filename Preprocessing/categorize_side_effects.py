import pandas as pd
import os
import re
 
def categorize_side_effects(csv_path):
    """
    Categorize side effects from multiple columns into 10 predefined categories,
    remove original side effect columns, and save as a new CSV file.
    
    Parameters:
    csv_path (str): Path to the CSV file containing side effect columns
    """
    # Define the categories and their associated side effects
    categories = {
        "Cardiovascular": [
            "Abnormal ECG", "Altered blood lipid level", "Angina", "Angina pectoris", "Arrhythmia",
            "Atrial arrhythmias", "Atrioventricular block", "Blood clot", "Blood clots",
            "Brain hemorrhage", "Cardiac arrest", "Cardiac failure", "Cardiovascular complications",
            "Change in blood pressure", "Chest discomfort", "Chest pain", "Chest tightness",
            "Circulatory disorder", "Congestive cardiac failure", "Cyanosis", "Decrease in blood pressure",
            "Decreased blood pressure", "Decreased cardiac function", "Dyslipidemia", "ECG changes",
            "Enlarged heart", "Extrasystoles", "Fast heart rate", "Flushing", "Flushing of face",
            "Heart attack", "Heart failure", "High blood pressure", "Hypertensive crisis",
            "Increased blood lipid level", "Increased heart rate", "Increased levels of blood fat",
            "Increased triglyceride level in blood", "Irregular heartbeats", "Ischemia",
            "Ischemic heart disease", "Myocardial infarction", "Orthostatic hypotension", "Palpitations",
            "Peripheral ischemia", "Postural hypotension", "Premature atrial contractions",
            "Procedural hypotension", "Prolonged QT interval", "Pulmonary hypertension",
            "Raynaud's phenomenon", "Reflex bradycardia", "Slow heart rate", "Systemic hypertension",
            "Tachycardia", "Vascular dilatation", "Vascular engorgement", "Vascular occlusion",
            "Vasculitis", "Vasoconstriction of the extremities", "Vasodilation",
            "Ventricular arrhythmia", "Ventricular premature contractions"
        ],
        "Gastrointestinal": [
            "Abdominal bloating", "Abdominal cramp", "Abdominal discomfort", "Abdominal distension",
            "Abdominal dysesthesia", "Abdominal pain", "Acid regurgitation", "Anal irritation",
            "Anal itching", "Anal ulcers", "Ascites", "Black colored stool", "Bloating",
            "Blood in stool", "Bowel incontinence", "Burping", "Change in bowel habits",
            "Chalky taste", "Clostridium difficile associated diarrhea", "Coating on tongue",
            "Constipation", "Dark colored stool", "Diarrhea", "Difficulty in swallowing",
            "Discoloration of stool", "Discoloration of teeth", "Dry lips", "Dry mouth",
            "Dry throat", "Dryness in mouth", "Dyspepsia", "Epigastric pain", "Esophageal bleeding",
            "Esophagitis", "Excessive salivation", "Fat in stool", "Fecal impaction", "Flatulence",
            "Flatus with discharge", "Gallstones", "Gastric irritation", "Gastritis",
            "Gastro-esophageal reflux disease", "Gastroenteritis", "Gastrointesinal symptoms",
            "Gastrointestinal bleeding", "Gastrointestinal discomfort", "Gastrointestinal disorder",
            "Gastrointestinal disturbance", "Gastrointestinal inflammation", "Gastrointestinal irritation",
            "Gastrointestinal motility disorder", "Gastrointestinal perforation", "Gastrointestinal toxicity",
            "Gastrointestinal ulcer", "Gingival hypertrophy", "Glossitis", "Gum Irritation",
            "Gum swelling", "Hard Dental Plaque", "Heartburn", "Hemorrhagic colitis",
            "Increased bowel movements", "Increased gastric acid secretion", "Increased saliva production",
            "Increased urge to pass bowels", "Indigestion", "Intestinal ulcer", "Lip inflammation",
            "Lip swelling", "Lower abdominal pain", "Mouth sore", "Mouth swelling", "Mouth ulcer",
            "Mucosal inflammation", "Nausea", "Numbness in mouth", "Oily evacuation", "Oily spotting",
            "Oral hypoesthesia", "Oral infections", "Oral ulcer", "Oropharyngeal pain",
            "Pancreatic inflammation", "Peptic ulcer", "Pharyngeal edema", "Pharyngeal hypoesthesia",
            "Pharyngeal pain", "Pseudomembranous colitis", "Rectal bleeding", "Rectal discomfort",
            "Retching", "Salivary gland inflammation", "Salivation", "Soft stools", "Sore throat",
            "Sore tongue", "Soreness", "Staining of teeth", "Sticky stools", "Stomach cramp",
            "Stomach discomfort", "Stomach fullness", "Stomach inflammation", "Stomach irritation",
            "Stomach pain", "Stomach pain/epigastric pain", "Stomach pressure", "Stomatitis",
            "Taste change", "Tenesmus", "Throat infection", "Throat irritation", "Throat pain",
            "Thrush", "Tooth discolouration", "Tongue discoloration", "Tongue irritation",
            "Tongue pain", "Ulcer", "Unpleasant breath odour", "Upper abdominal pain",
            "Upset stomach", "Vomiting", "Watery diarrhoea", "Xerostomia"
        ],
        "Dermatological": [
            "Acne", "Acne-like rash", "Acneiform dermatitis", "Acneiform eruptions",
            "Allergic contact dermatitis", "Allergic dermatitis", "Allergic skin rash",
            "Angioedema", "Argyria", "Blisters", "Blisters on fingers/feet", "Blisters on skin",
            "Bruise", "Bruising", "Burning sensation", "Calcinosis cutis", "Changes in skin color",
            "Cold skin", "Cold sores", "Contact dermatitis", "Dermatitis", "Discoloration of skin",
            "Dry skin", "Drug eruptions", "Drug rash with eosinophilia and systemic symptoms",
            "Ecchymosis", "Eczema", "Eczematoid dermatitis", "Erythema", "Erythema multiforme",
            "Erythematous rash", "Excessive hair growth on face", "Facial hair growth",
            "Facial swelling", "Flakes on scalp", "Hair discoloration", "Hair follicle inflammation",
            "Hair loss", "Hand-foot syndrome", "Hives", "Hyperpigmentation", "Hypopigmentation",
            "Increased facial sebum production", "Increased hair growth", "Inflammation of hair follicles",
            "Itching", "Itchy scalp", "Lipodystrophy", "Maculopapular rash", "Miliaria", "Oily skin",
            "Pale red skin", "Pale skin", "Palmar-plantar erythrodysesthesia syndrome", "Papular rash",
            "Perioral dermatitis", "Petechiae", "Photosensitivity", "Phototoxicity", "Pruritis",
            "Psoriasis", "Purpura", "Rash", "Redness of skin", "Red spots or bumps", "Ring scotoma",
            "Rosacea", "Scaling", "Scratch dermatitis", "Skin and cutaneous disorders", "Skin atrophy",
            "Skin bleeding", "Skin burn", "Skin discoloration", "Skin disorder", "Skin erosion",
            "Skin eruptions", "Skin exfoliation", "Skin flakes", "Skin irritation", "Skin lesion",
            "Skin maceration", "Skin pain", "Skin peeling", "Skin pigmentation", "Skin rash",
            "Skin reaction", "Skin staining", "Skin swelling", "Skin ulcer", "Stevens-Johnson syndrome",
            "Stretch marks", "Sweating", "Telangiectasia", "Thinning of skin", "Toxic epidermal necrolysis",
            "Transient stinging", "Urticaria", "Vesicular dermatitis"
        ],
        "Central Nervous System": [
            "Abnormal behavior", "Abnormal dreams", "Abnormal gait", "Abnormal sensation",
            "Abnormal thoughts", "Affect lability", "Aggravation of depression", "Aggression",
            "Aggressive behavior", "Agitation", "Akathisia", "Altered walking", "Anger", "Anxiety",
            "Apnea", "Balance disorder", "Behavioral changes", "Brain swelling", "CNS stimulation",
            "CNS toxicity", "Central nervous system depression", "Cognitive disorder", "Coma",
            "Confusion", "Convulsion", "Coordination disorder", "Delirium", "Delusion", "Depression",
            "Difficulty in paying attention", "Difficulty in speaking", "Disorientation", "Dizziness",
            "Drowsiness", "Dysphoria", "Dystonia", "Emotional lability", "Euphoria", "Excitation",
            "Excitement", "Extrapyramidal symptoms", "Fainting", "Fear", "Flaccid paralysis",
            "Generalized tonic-clonic seizure", "Hallucination", "Headache", "Hemiparesis",
            "Hemiplegia", "Hostility", "Hyperactivity", "Hypertonia", "Hypoesthesia", "Hypomania",
            "Hypotonia", "Idiopathic intracranial hypertension", "Impaired concentration",
            "Impaired coordination", "Impaired judgment", "Inability to concentrate",
            "Increased alertness", "Increased intracranial pressure", "Insomnia",
            "Involuntary muscle movement", "Irritability", "Lethargy", "Lightheadedness",
            "Loss of accommodation", "Low energy", "Mania", "Memory impairment", "Memory loss",
            "Mental impairment", "Migraine", "Mood changes", "Mood swings", "Muscle coordination impaired",
            "Myasthenia gravis", "Nervousness", "Neuritis", "Neuropathy", "Neuropsychiatric symptoms",
            "Neurotoxicity", "Nightmares", "Numbness", "Nystagmus", "Paranoia", "Paresthesia",
            "Parkinson-like symptoms", "Parkinsonism", "Parosmia", "Peripheral neuropathy",
            "Psychomotor impairment", "Psychosis", "Psychotic disorder", "Restlessness", "Rigidity",
            "Seizure", "Shakiness", "Sleep disorder", "Sleep disturbance", "Sleepiness",
            "Slurred speech", "Speech disorder", "Suicidal behaviors", "Tightness sensation",
            "Tingling", "Tingling sensation", "Tingling sensation of extremity", "Tiredness",
            "Tonic-clonic seizures", "Trembling", "Tremors", "Uncoordinated body movements",
            "Unsteadiness", "Vertigo", "Walking difficulties", "Weakness", "Yawning"
        ],
        "Allergic and Immunological": [
            "Allergic conjunctivitis", "Allergic reaction", "Allergic reaction in eye",
            "Allergic sensitization", "Allergy", "Anaphylactic reaction", "Auto-antibody formation",
            "Conjunctival sensitivity", "Cytokine release syndrome", "Decreased immunity",
            "Decreased immunoglobulins in infants", "Drug fever", "Febrile response", "Hay fever",
            "Hypersensitivity", "Increased risk of infection", "Opportunistic infections",
            "Secondary infections", "Serum sickness", "Systemic lupus erythematosus"
        ],
        "Hematological": [
            "Abnormal blood cell count", "Abnormal bruising", "Agranulocytosis",
            "Altered blood sugar level", "Anemia", "Aplastic anemia", "Azotemia", "Bleeding",
            "Bleeding disorder", "Bleeding under the skin", "Blood disorder", "Blood dyscrasias",
            "Bone marrow failure", "Bone marrow suppression", "Cyanide poisoning",
            "Decreased blood cells", "Decreased hematocrit level", "Decreased hemoglobin",
            "Decreased white blood cell count", "Decreased white blood cell count (lymphocytes)",
            "Decreased white blood cell count (neutrophils)", "Easy bruise", "Febrile neutropenia",
            "Fever due to low white blood cell count", "Glucose intolerance", "Granulocytopenia",
            "Hematological disorder", "Hematoma", "Hemolytic anemia", "Hemorrhage",
            "Hemorrhagic complications", "High white blood cell count", "Hypoglycaemia",
            "Hypoglycemia", "Idiopathic thrombocytopenic purpura", "Increase haematocrit",
            "Increased bilirubin in the blood", "Increased bleeding tendency", "Increased bleeding time",
            "Increased glucose level in blood", "Increased hematocrit", "Increased hemoglobin",
            "Increased prothrombin time", "Increased red blood cells", "Increased white blood cell count",
            "Increased white blood cell count (eosinophils)", "Low albumin level in blood",
            "Low blood platelets", "Lymphocytopenia", "Lymphopenia", "Myelodysplastic syndrome",
            "Myelosuppression", "Negative nitrogen balance", "Oligospermia",
            "Prolonged activated partial thromboplastin time", "Prolonged bleeding", "Sepsis",
            "Thrombocytosis", "Thromboembolism", "Thrombophlebitis", "Thrombotic thrombocytopenic purpura",
            "Transient increase in white blood cell count"
        ],
        "Musculoskeletal": [
            "Abnormal involuntary movements", "Abnormality of voluntary movements", "Back pain",
            "Bone fracture", "Bone pain", "Bone weakness", "Compression fracture",
            "Connective tissue disorders", "Cramps", "Fracture", "Gout", "Gout flares",
            "Inflammation of tendons", "Jaw pain", "Joint inflammation", "Joint pain",
            "Joint stiffness", "Joint swelling", "Leg cramps", "Leg pain", "Limb pain",
            "Muscle contraction", "Muscle cramp", "Muscle damage", "Muscle disorders",
            "Muscle pain", "Muscle rigidity", "Muscle spasm", "Muscle stiffness", "Muscle twitching",
            "Muscle weakness", "Musculoskeletal pain", "Myositis", "Neck pain", "Nerve pain",
            "Osteoporosis", "Pain", "Pain in extremities", "Pain in extremity",
            "Postoperative muscle pain", "Reduced bone density", "Softening of bones",
            "Steroid myopathy", "Stiffness", "Tendon rupture"
        ],
        "Respiratory": [
            "Airway inflammation", "Asthma", "Breathing problems", "Breathlessness", "Bronchitis",
            "Bronchoconstriction", "Bronchospasm", "Chest congestion", "Chronic lung disease",
            "Chronic obstructive pulmonary disease", "Cold symptoms", "Common cold", "Cough",
            "Coughing up blood", "Decreased pulmonary function", "Dry cough", "Dry nose", "Flu",
            "Flu-like symptoms", "Hoarseness of voice", "Hypoxia", "Increased bronchial secretions",
            "Increased respiratory rate", "Increased sputum production", "Influenza",
            "Interstitial pneumonia", "Laryngeal edema", "Lung damage", "Lung disorder",
            "Lung infection", "Nasal congestion", "Nasal discomfort", "Nasal dryness",
            "Nasal inflammation", "Nasal irritation", "Nasal ulceration", "Nasopharyngeal irritation",
            "Nasopharyngitis", "Nosebleeds", "Pneumonia", "Pneumothorax", "Pulmonary fibrosis",
            "Pulmonary hemorrhage", "Reduce bronchial secretion", "Reduced bronchial secretions",
            "Respiratory arrest", "Respiratory depression", "Respiratory disorder",
            "Respiratory tract infection", "Respiratory tract inflammation", "Runny nose",
            "Shortness of breath", "Sinus disorder", "Sinus infection", "Sinus inflammation",
            "Sinus pain", "Sneezing", "Thickened respiratory tract secretions",
            "Upper respiratory tract infection", "Voice change", "Wheezing",
            "Worsening of a pre-existing breathing problem"
        ],
        "Urinary and Renal": [
            "Abnormal kidney function test", "Abnormal renal function tests", "Abnormal urination",
            "Acute renal failure", "Altered frequency of urination", "Azotemia", "Bilirubin in urine",
            "Blood in urine", "Chromaturia", "Decreased creatinine clearance", "Dark colored urine",
            "Dark yellow to brown discoloration of urine", "Decreased protein levels in blood",
            "Difficulty in urination", "Discomfort when urinating", "Frequent urge to urinate",
            "Glucose in urine", "Glycosuria", "Increased calcium in urine", "Increased production of urine",
            "Increased uric acid level in urine", "Irritable bladder", "Ketones in urine",
            "Kidney damage", "Kidney stone", "Low urine output", "Micturition disorders",
            "Nephritic syndrome", "Nephritis", "Nephrocalcinosis", "Nephrotic syndrome", "Nocturia",
            "Polyuria", "Protein in urine", "Renal dysfunction", "Renal impairment", "Renal injury",
            "Renal tubular acidosis", "Urinary incontinence", "Urinary retention",
            "Urinary tract disorder", "Urinary tract infection", "Urine discoloration", "Urolithiasis"
        ],
        "Endocrine and Metabolic": [
            "Abnormal phosphorous level in blood", "Adrenal insufficiency", "Altered blood sugar level",
            "Carbohydrate intolerance", "Cushing syndrome", "Cushingoid syndrome",
            "Decreased calcium level in blood", "Decreased cholesterol level in blood",
            "Decreased copper level", "Decreased level of thyroid hormones",
            "Decreased magnesium level in blood", "Decreased phosphate level in blood",
            "Decreased potassium level in blood", "Decreased sodium level in blood", "Diabetes",
            "Electrolyte imbalance", "Elevated creatine kinase", "Elevated creatinine kinase",
            "Elevated levels of serum amylase", "Elevated serum glutamic oxaloacetic transaminase",
            "Elevated serum glutamic pyruvic transaminase", "Folic acid deficiency", "Goiter",
            "Growth retardation", "Growth retardation in children", "Hormone imbalance",
            "Hypercholesterolemia", "Hypereosinophilia", "Hypernatremia", "Hyperthyroidism",
            "Hypokalemic alkalosis", "Hyponatraemia", "Hypothalamic-pituitary-adrenal axis suppression",
            "Hypothyroidism", "Increased alkaline phosphatase level in blood", "Increased androgen levels",
            "Increased appetite", "Increased aspartate aminotransferase", "Increased blood urea",
            "Increased blood urea nitrogen", "Increased blood uric acid", "Increased calcium level in blood",
            "Increased creatine phosphokinase level in blood", "Increased creatinine level in blood",
            "Increased gamma-glutamyltransferase", "Increased lactate dehydrogenase level in blood",
            "Increased lipase in the blood", "Increased liver enzymes", "Increased phosphate level in blood",
            "Increased potassium level in blood", "Increased prolactin level in blood", "Increased thirst",
            "Increased transaminase level in blood", "Increased uric acid level in blood",
            "Iron deficiency", "Lactic acidosis", "Latent diabetes mellitus", "Loss of appetite",
            "Metabolic acidosis", "Metabolic alkalosis", "Metabolic disorder", "Obesity",
            "Precocious puberty", "Sodium retention", "Thyroid disorders", "Volume depletion",
            "Weight gain", "Weight loss"
        ]
    }
    
    # Load the CSV file
    print(f"Loading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Identify side effect columns
    side_effect_cols = [col for col in df.columns if re.match(r'sideEffect\d+', col)]
    print(f"Found {len(side_effect_cols)} side effect columns: {side_effect_cols}")
    
    # Create a flattened list of all side effects for easier matching
    all_side_effects = []
    for category, effects in categories.items():
        all_side_effects.extend([(effect.lower(), category) for effect in effects])
    
    # Create a mapping dictionary for quick lookup
    side_effect_to_category = {effect: category for effect, category in all_side_effects}
    
    # Initialize dictionaries to collect side effects for each category
    categorized_side_effects = {category: [] for category in categories.keys()}
    
    # Process each row
    print("Categorizing side effects...")
    for index, row in df.iterrows():
        if index % 1000 == 0 and index > 0:
            print(f"Processed {index} rows...")
            
        # Reset category lists for this row
        row_categories = {category: [] for category in categories.keys()}
        
        # For each side effect column in the row
        for col in side_effect_cols:
            side_effect = str(row[col]).lower()
            if pd.notna(row[col]) and side_effect != 'nan' and side_effect != '':
                # Try to match the side effect to a category
                category_found = False
                
                # First try direct match
                if side_effect in side_effect_to_category:
                    category = side_effect_to_category[side_effect]
                    row_categories[category].append(row[col])  # Use original case from CSV
                    category_found = True
                else:
                    # Try partial match if direct match fails
                    for effect, category in all_side_effects:
                        if (effect in side_effect) or (side_effect in effect):
                            row_categories[category].append(row[col])  # Use original case from CSV
                            category_found = True
                            break
                
                # If no match found, check for keywords
                if not category_found:
                    # Common keywords for each category
                    if any(kw in side_effect for kw in ["heart", "blood pressure", "cardiac", "chest pain", "pulse"]):
                        row_categories["Cardiovascular"].append(row[col])
                    elif any(kw in side_effect for kw in ["stomach", "intestine", "bowel", "diarrhea", "nausea", "vomit", "digest"]):
                        row_categories["Gastrointestinal"].append(row[col])
                    elif any(kw in side_effect for kw in ["skin", "rash", "itch", "derma"]):
                        row_categories["Dermatological"].append(row[col])
                    elif any(kw in side_effect for kw in ["brain", "dizzy", "headache", "nervous", "mood", "depress", "sleep", "anxiety"]):
                        row_categories["Central Nervous System"].append(row[col])
                    elif any(kw in side_effect for kw in ["allerg", "immune", "sensitiv"]):
                        row_categories["Allergic and Immunological"].append(row[col])
                    elif any(kw in side_effect for kw in ["blood", "platelet", "anemia", "bleed"]):
                        row_categories["Hematological"].append(row[col])
                    elif any(kw in side_effect for kw in ["muscle", "bone", "joint", "pain", "cramp"]):
                        row_categories["Musculoskeletal"].append(row[col])
                    elif any(kw in side_effect for kw in ["breath", "lung", "respir", "cough", "nose", "sinus"]):
                        row_categories["Respiratory"].append(row[col])
                    elif any(kw in side_effect for kw in ["kidney", "urin", "bladder", "renal"]):
                        row_categories["Urinary and Renal"].append(row[col])
                    elif any(kw in side_effect for kw in ["hormone", "thyroid", "metabol", "weight", "appetite", "diabetes"]):
                        row_categories["Endocrine and Metabolic"].append(row[col])
        
        # Add categorized side effects to the dataframe
        for category, effects in row_categories.items():
            if effects:  # Only add if there are effects in this category
                df.at[index, category] = ", ".join(effects)
            else:
                df.at[index, category] = ""
    
    # Drop the original side effect columns
    df = df.drop(columns=side_effect_cols)
    
    # Create output filename based on input filename
    base_name = os.path.basename(csv_path)
    file_name, file_ext = os.path.splitext(base_name)
    output_path = os.path.join(os.path.dirname(csv_path), f"{file_name}_categorized{file_ext}")
    
    # Save the modified dataframe
    print(f"Saving categorized data to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Done!")
    
    return output_path
 
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        categorize_side_effects(csv_path)
    else:
        print("Usage: python categorize_side_effects.py <path_to_csv_file>")