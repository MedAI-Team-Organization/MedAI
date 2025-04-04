import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# Step 1: Load and Explore Data
def load_and_explore_data(filepath):
    """Load the medicine dataset and perform initial exploration"""
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    return df

# Step 2: Preprocess Data
def preprocess_data(df):
    """Preprocess the data for modeling"""
    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':        
            df[col] = df[col].fillna('')
        else:
            df[col] = df[col].fillna(0)
    
    # Process comma-separated values
    for col in ['substitutes', 'usecases', 'composition', 'sideeffects']:
        df[col] = df[col].astype(str)
    
    # Create a composite text feature from all input features
    df['text_features'] = df['name'] + ' ' + \
                          df['substitutes'] + ' ' + \
                          df['usecases'] + ' ' + \
                          df['Chemical Class'] + ' ' + \
                          df['Habit Forming'] + ' ' + \
                          df['Therapeutic Class'] + ' ' + \
                          df['Action Class'] + ' ' + \
                          df['manufacturer_name'] + ' ' + \
                          df['composition']
    
    # Extract side effects as a list for each medicine
    df['side_effects_list'] = df['sideeffects'].apply(lambda x: [effect.strip() for effect in str(x).split(',') if effect.strip()])
    
    return df

# Step 3: Feature Engineering
def engineer_features(df, max_features=5000):
    """Convert text features to TF-IDF representation"""
    # Text feature vectorization
    vectorizer = TfidfVectorizer(max_features=max_features, 
                                 stop_words='english',
                                 ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['text_features'])
    
    # Process target variable
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['side_effects_list'])
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")
    print(f"Number of unique sideeffects: {len(mlb.classes_)}")
    
    return X, y, vectorizer, mlb

# Step 4: Split Data
def split_dataset(X, y):
    """Split data into training, validation, and test sets (60%, 20%, 20%)"""
    # First split: 80% train+val, 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Second split: 60% train, 20% validation (from the 80%)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42) # 0.25 of 80% is 20%
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Step 5: Train Model
def train_model(X_train, y_train, X_val, y_val):
    """Train a multi-label classification model"""
    # Using Random Forest as base classifier
    base_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    
    # Multi-output classifier for multi-label classification
    model = MultiOutputClassifier(base_clf)
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_hamming_loss = hamming_loss(y_val, y_val_pred)
    
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Hamming Loss: {val_hamming_loss:.4f}")
    
    return model

# Step 6: Evaluate Model
def evaluate_model(model, X_test, y_test, mlb):
    """Evaluate the model on test data"""
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_hamming_loss = hamming_loss(y_test, y_pred)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Hamming Loss: {test_hamming_loss:.4f}")
    
    # Get classification report for each side effect (for top 10 most common side effects)
    top_side_effects = np.argsort(np.sum(y_test, axis=0))[-10:]
    
    for i in top_side_effects:
        side_effect = mlb.classes_[i]
        print(f"\nClassification Report for '{side_effect}':")
        print(classification_report(y_test[:, i], y_pred[:, i]))
    
    return y_pred, test_accuracy, test_hamming_loss

# Step 7: Visualize Results
def visualize_results(y_test, y_pred, mlb):
    """Visualize model performance"""
    # Calculate frequency of each side effect
    side_effect_counts = np.sum(y_test, axis=0)
    
    # Get top 15 most common side effects
    top_indices = np.argsort(side_effect_counts)[-15:]
    top_side_effects = mlb.classes_[top_indices]
    
    # Calculate F1 score for each side effect
    f1_scores = []
    from sklearn.metrics import f1_score
    
    for i in top_indices:
        f1 = f1_score(y_test[:, i], y_pred[:, i])
        f1_scores.append(f1)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Sort for better visualization
    sorted_indices = np.argsort(f1_scores)
    sorted_side_effects = [top_side_effects[i] for i in sorted_indices]
    sorted_f1_scores = [f1_scores[i] for i in sorted_indices]
    
    sns.barplot(x=sorted_f1_scores, y=sorted_side_effects)
    plt.title('F1 Score for Top 15 Most Common Side Effects')
    plt.xlabel('F1 Score')
    plt.tight_layout()
    plt.savefig('side_effect_f1_scores.png')
    plt.show()
    
    # Confusion matrix for top side effect
    top_effect_idx = np.argmax(side_effect_counts)
    top_effect = mlb.classes_[top_effect_idx]
    
    print(f"Confusion Matrix for most common side effect: '{top_effect}'")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test[:, top_effect_idx], y_pred[:, top_effect_idx])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {top_effect}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    plt.show()

# Step 8: Function to predict side effects for new medicines
def predict_side_effects(medicine_data, vectorizer, model, mlb):
    """Predict side effects for a new medicine"""
    # Create text features
    text_features = medicine_data['name'] + ' ' + \
                   medicine_data['substitutes'] + ' ' + \
                   medicine_data['usecases'] + ' ' + \
                   medicine_data['Chemical Class'] + ' ' + \
                   medicine_data['Habit Forming'] + ' ' + \
                   medicine_data['Therapeutic Class'] + ' ' + \
                   medicine_data['Action Class'] + ' ' + \
                   medicine_data['manufacturer_name'] + ' ' + \
                   medicine_data['composition']
    
    # Vectorize
    X_new = vectorizer.transform([text_features])
    
    # Predict
    y_pred = model.predict(X_new)
    
    # Get side effect names
    predicted_indices = np.where(y_pred[0] == 1)[0]
    predicted_side_effects = mlb.classes_[predicted_indices]
    
    return predicted_side_effects

# Main execution
def main():
    # Load data
    df = load_and_explore_data('merged_file.csv')
    
    # Preprocess
    df = preprocess_data(df)
    
    # Engineer features
    X, y, vectorizer, mlb = engineer_features(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    y_pred, test_accuracy, test_hamming_loss = evaluate_model(model, X_test, y_test, mlb)
    
    # Visualize results
    visualize_results(y_test, y_pred, mlb)
    
    # Save the model and vectorizer
    model_path = r"C:\Users\jignesh01\OneDrive - Conestoga College\side_effect_model.joblib"
    vectorizer_path = r"C:\Users\jignesh01\OneDrive - Conestoga College\tfidf_vectorizer.joblib"
    mlb_path = r"C:\Users\jignesh01\OneDrive - Conestoga College\mlb.joblib"
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(mlb, mlb_path)
    
    print(f"Model saved at {model_path}")
    print(f"TF-IDF Vectorizer saved at {vectorizer_path}")
    print(f"MultiLabelBinarizer saved at {mlb_path}")

# Run main
if __name__ == "__main__":
    main()
