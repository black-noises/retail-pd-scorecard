"""
Retail Customer CD Scorecard - Python Script Version
A simple Probability of Default (PD) scorecard for retail customers using Random Forest.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def generate_customer_data(n_customers=1000, random_seed=42):
    """Generate synthetic retail customer data"""
    np.random.seed(random_seed)
    
    data = {
        'customer_id': range(1, n_customers + 1),
        'age': np.random.normal(40, 12, n_customers).astype(int),
        'income': np.random.lognormal(10.5, 0.5, n_customers).astype(int),
        'credit_score': np.random.normal(650, 100, n_customers).astype(int),
        'employment_length': np.random.exponential(5, n_customers).astype(int),
        'debt_to_income': np.random.beta(2, 5, n_customers),
        'num_credit_cards': np.random.poisson(2, n_customers),
        'has_mortgage': np.random.choice([0, 1], n_customers, p=[0.4, 0.6]),
        'num_late_payments': np.random.poisson(1, n_customers),
        'credit_utilization': np.random.beta(3, 2, n_customers)
    }
    
    df = pd.DataFrame(data)
    
    # Cap values at realistic ranges
    df['age'] = df['age'].clip(18, 80)
    df['credit_score'] = df['credit_score'].clip(300, 850)
    df['debt_to_income'] = df['debt_to_income'].clip(0, 1)
    df['credit_utilization'] = df['credit_utilization'].clip(0, 1)
    
    return df

def calculate_default_probability(row):
    """Calculate default probability based on risk factors"""
    prob = 0.05  # 5% base default rate
    
    # Credit score impact
    if row['credit_score'] < 500:
        prob += 0.25
    elif row['credit_score'] < 600:
        prob += 0.15
    elif row['credit_score'] < 700:
        prob += 0.05
    
    # Debt to income ratio impact
    if row['debt_to_income'] > 0.5:
        prob += 0.15
    elif row['debt_to_income'] > 0.3:
        prob += 0.05
    
    # Late payments impact
    if row['num_late_payments'] > 3:
        prob += 0.20
    elif row['num_late_payments'] > 1:
        prob += 0.10
    
    # Credit utilization impact
    if row['credit_utilization'] > 0.8:
        prob += 0.15
    elif row['credit_utilization'] > 0.5:
        prob += 0.05
    
    # Income factor
    if row['income'] < 30000:
        prob += 0.10
    
    return min(prob, 0.95)

def create_target_variable(df):
    """Create target variable (default)"""
    df['default_probability'] = df.apply(calculate_default_probability, axis=1)
    df['defaulted'] = (np.random.random(len(df)) < df['default_probability']).astype(int)
    return df

def train_model(df):
    """Train Random Forest model"""
    feature_columns = [
        'age', 'income', 'credit_score', 'employment_length',
        'debt_to_income', 'num_credit_cards', 'has_mortgage',
        'num_late_payments', 'credit_utilization'
    ]
    
    X = df[feature_columns]
    y = df['defaulted']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    
    return rf_model, X_train, X_test, y_train, y_test, feature_columns

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC Score: {auc_score:.3f}")
    
    return y_pred, y_pred_proba

def plot_feature_importance(model, feature_columns):
    """Plot feature importance"""
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance in Random Forest Model')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
    
    return feature_importance

def calculate_credit_score(model, customer_data):
    """Calculate credit score for a customer"""
    default_prob = model.predict_proba(customer_data)[0, 1]
    credit_score = int(850 - (default_prob * 550))
    credit_score = max(300, min(850, credit_score))
    return credit_score, default_prob

def main():
    """Main execution function"""
    print("🏦 Retail Customer PD Scorecard")
    print("=" * 50)
    
    # Generate data
    print("📊 Generating customer data...")
    df = generate_customer_data()
    df = create_target_variable(df)
    print(f"Generated {len(df)} customer records")
    print(f"Default rate: {df['defaulted'].mean():.2%}")
    
    # Train model
    print("\n🤖 Training Random Forest model...")
    model, X_train, X_test, y_train, y_test, features = train_model(df)
    print(f"Model trained successfully!")
    print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
    print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Feature importance
    print("\n🎯 Feature Importance:")
    importance_df = plot_feature_importance(model, features)
    print(importance_df)
    
    # Sample predictions
    print("\n💳 Sample Customer Scorecards:")
    print("=" * 50)
    sample_customers = X_test.head(3)
    
    for idx, (i, customer) in enumerate(sample_customers.iterrows()):
        credit_score, default_prob = calculate_credit_score(model, customer.values.reshape(1, -1))
        actual_default = y_test.iloc[i]
        
        print(f"\nCustomer {idx+1}:")
        print(f"  Credit Score: {credit_score}")
        print(f"  Default Probability: {default_prob:.2%}")
        print(f"  Actual Default: {'Yes' if actual_default else 'No'}")
        print(f"  Key Features: Credit Score={customer['credit_score']:.0f}, DTI={customer['debt_to_income']:.2f}")
    
    # Save model
    print("\n💾 Saving model...")
    joblib.dump(model, 'random_forest_scorecard_model.pkl')
    df.to_csv('retail_customer_data.csv', index=False)
    print("Model and data saved successfully!")
    
    print("\n✅ Scorecard creation complete!")

if __name__ == "__main__":
    main()
