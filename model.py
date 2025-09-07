import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data(filepath):
    """Load the CSV file into a pandas DataFrame.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    # TODO: Load the data from the CSV file and return as a DataFrame
    df = pd.read_csv(filepath)
    return df

def check_missing_values(df):
    """Return the number of missing values in each column.
    
    Returns:
        pd.Series: Index = column names, Values = number of missing entries.
    """
    # TODO: Implement missing values check and return the result
    result = df.isnull().sum()
    return result

def churn_balance(df):
    """Return churn distribution statistics.
    
    Returns:
        dict: {
            "total": int,         # total number of customers
            "churned": int,       # number of churned customers
            "non_churned": int,   # number of non-churned customers
            "churn_rate": float   # churn rate (between 0 and 1)
        }
    """
    # TODO: Calculate churn statistics and return as a dictionary
    total = len(df)
    churned = df["Exited"].sum()
    non_churned = total - churned
    churn_rate = churned/total if total > 0 else 0.0

    result = {
        "total" : total,
        "churned" : churned,
        "non_churned" : non_churned,
        "churn_rate" : churn_rate
    }
    return result

def descriptive_statistics(df):
    """Return descriptive statistics of numeric columns.
    
    Returns:
        pd.DataFrame: Statistical summary (mean, std, min, max, etc.)
    """
    # TODO: Compute descriptive statistics and return the DataFrame
    result = df.describe()
    return result

def remove_irrelevant_columns(df):
    """
    Remove columns RowNumber, CustomerId, Surname from the dataframe.
    """
    # TODO: Drop the irrelevant columns and return the modified dataframe
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1)
    return df

def encode_categorical(df):
    """
    Encode categorical columns (Geography, Gender) using one-hot encoding.
    """
    # TODO: Use one-hot encoding to convert the categorical columns into numeric form
    df = pd.get_dummies(df, columns = ['Geography', 'Gender'], drop_first = True)
    return df

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the dataset into features (X) and target (y), then into training and test sets.
    """
    # TODO: Separate 'Exited' column as target and split data into X_train, X_test, y_train, y_test
    X = df.drop('Exited', axis = 1)
    y = df['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Scale numerical features in X_train and X_test using StandardScaler.
    """
    # TODO: Fit a StandardScaler on X_train and transform X_train and X_test. Return the scaled data and the scaler.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def build_model():
    """
    Build and return the MLPClassifier model with the following configuration:
    - Two hidden layers: 1st with 64 neurons and 2nd with 32 neurons
    - ReLU activation function for hidden layers
    - Adam optimizer for stochastic gradient-based optimization
    - Learning rate: 0.01
    - Early stopping enabled
    - Maximum of 100 iterations (epochs) for training
    - validation_fraction = 0.2
    - n_iter_no_change = 10 (Stops training early if validation score does not improve for 10 consecutive epochs)
    - alpha = 0.001
    """
    # TODO: Create and return an MLPClassifier with the given hyperparameters
    model = MLPClassifier(
        hidden_layer_sizes = (64, 32),
        activation = 'relu',
        solver = 'adam',
        learning_rate_init = 0.01,
        early_stopping = True,
        max_iter = 100,
        validation_fraction = 0.2,
        n_iter_no_change = 10,
        alpha = 0.001,
        random_state = 42
    )

    return model

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Train the model, evaluate it on the test data, and print metrics.
    """
    # TODO: Fit the model on training data, predict on test data, and print accuracy and classification report
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    class_rep = classification_report(y_test, y_pred)

    print(acc)
    print(class_rep)

    return model

def save_object(obj, filename):
    """
    Save a Python object to a file using pickle.
    """
    # TODO: Save the provided object to disk using pickle
    with open(filename, "wb") as f:
        pickle.dump(obj, f)



if __name__ == "__main__":
    # Load the dataset
    df = load_data("Churn_Modelling.csv")
    
    # Get missing values
    missing_values = check_missing_values(df)
    print("Missing values in each column:")
    print(missing_values)

    print("\n")
    
    # Get churn distribution
    churn_stats = churn_balance(df)
    print(f"Total customers: {churn_stats['total']}")
    print(f"Churned customers: {churn_stats['churned']}")
    print(f"Stayed customers: {churn_stats['non_churned']}")
    print(f"Churn rate: {churn_stats['churn_rate']:.2%}")

    print("\n")
    
    # Get descriptive statistics
    stats = descriptive_statistics(df)
    print("Descriptive statistics for numerical features:")
    print(stats)

    df = remove_irrelevant_columns(df)
    df = encode_categorical(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print(df.head(5))

    model = build_model()
    model = train_and_evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test)

    save_object(scaler, "scaler.pkl")
    save_object(model, "model.pkl")
    print("Model and scaler saved successfully.")