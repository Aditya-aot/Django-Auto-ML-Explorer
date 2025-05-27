from django.shortcuts import render ,redirect
# Create your views here.
import os
import pandas as pd
from django.http import HttpResponse, FileResponse
from django.conf import settings
import tempfile
from django.views.decorators.csrf import csrf_exempt

def home(request) :

    return render(request, 'main_page/index.html')




def functions(request) :

    return render(request, 'main_page/fun.html')

def data_in(request) :

    return render(request, 'main_page/dataset input.html')


def data_analysis_in(request) :

    return render(request, 'main_page/analysis_input.html')





import pandas as pd
import sweetviz as sv
import os
import glob
import requests
def analysis_dashboard(request) :
    if request.method == 'POST' and request.FILES.get('csv_file'):
        uploaded_file = request.FILES['csv_file']
        file_name = uploaded_file.name

        # Save file to local project directory
        save_path = os.path.join(os.path.dirname(__file__), file_name)
        with open(save_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        try:
            df = pd.read_csv(save_path, encoding='ISO-8859-1', on_bad_lines='skip')
        except Exception as e:
            return HttpResponse(f"Could not decode file. Error: {str(e)}", status=400)


        df = pd.read_csv(save_path)
        print("Saving file to:", save_path)
        print("Uploaded file name:", file_name) 

        print(df.head())  # For debug

        column_names = df.columns.tolist()
        # above is to get csv data after saving it 
        ######################################################################################################
        ######################################################################################################
        # below is code for  Eda using sweet wiz (Exploratory Data Analysis)
        print(" outsite try ------------------------------------")
        try:
            print(" insite try ------------------------------------",column_names)
            df = pd.read_csv(save_path)

            # Features to use
            features = column_names
            df_selected = df[features]

            # Create Sweetviz report without opening it automatically
            report = sv.analyze(df_selected)
            report.show_html('car_price_analysis_report.html', open_browser=False)

            # Create output directory
            # output_dir = "csv Data Analysis"
            output_dir = "C:/Users/adity/projects/dataset_analysis/static/csv/csv Data Analysis"
            os.makedirs(output_dir, exist_ok=True)

            # 1. Summary statistics
            summary_stats = df_selected.describe(include='all')
            summary_stats.to_csv(os.path.join(output_dir, "summary_statistics.csv"))

            # 2. Missing values
            missing_values = df_selected.isnull().sum()
            missing_values.to_csv(os.path.join(output_dir, "missing_values.csv"), header=["MissingCount"])

            # 3. Missing percentage
            missing_percent = (df_selected.isnull().sum() / len(df_selected)) * 100
            missing_percent.to_csv(os.path.join(output_dir, "missing_percentage.csv"), header=["MissingPercentage"])

            # 4. Data types
            dtypes = df_selected.dtypes
            dtypes.to_csv(os.path.join(output_dir, "data_types.csv"), header=["DataType"])

            # 5. Correlation matrix
            correlation = df_selected.corr(numeric_only=True)
            correlation.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))

            # 6. Attribute counts for each feature
            attr_summary = []
            for col in df_selected.columns:
                value_counts = df_selected[col].value_counts().head(10)
                for val, count in value_counts.items():
                    attr_summary.append({
                        "Feature": col,
                        "Attribute": val,
                        "Count": count
                    })

            attr_df = pd.DataFrame(attr_summary)
            attr_df.to_csv(os.path.join(output_dir, "attribute_summary.csv"), index=False)

            print("All EDA CSV files saved in 'csv Data Analysis' folder.")

        except FileNotFoundError:
            print(f"Error: The file {save_path} was not found.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        
        ###############################################################################################
                # After all your EDA CSVs are saved
        try:
            # Path to your CSV data analysis folder
            analysis_dir = "C:/Users/adity/projects/dataset_analysis/static/csv/csv Data Analysis"

            # Collect all CSV content
            data_summary = {}
            for csv_file in ["summary_statistics.csv",
                            "correlation_matrix.csv", ]:
                path = os.path.join(analysis_dir, csv_file)
                if os.path.exists(path):
                    df_temp = pd.read_csv(path)
                    data_summary[csv_file] = df_temp.to_dict(orient='records')
                else:
                    data_summary[csv_file] = "File not found."

            # Convert collected data into prompt
            ollama_prompt = f"""
            You are a data analyst. Here are CSV summaries of a dataset. Please provide a comprehensive, readable explanation
            of what this data contains, insights on missing data, correlation observations, and potential data quality issues.
            Format your response as a structured summary.

            Data Summary:
            {json.dumps(data_summary, indent=2)}
            """

            # Call local Ollama API
            response = requests.post(
                            "http://localhost:11434/api/generate",
                            json={
                                "model": "mistral",
                                #  "model": "tinydolphin",
                                "prompt": ollama_prompt,
                                "stream": False
                            }
                        )


            result = response.json()
            ollama_text = result.get("response", "Ollama returned no response.")

            # Save Ollama analysis
            with open(os.path.join(analysis_dir, "ollama_analyse.csv"), "w", encoding="utf-8") as f:
                f.write("Insight\n")
                for line in ollama_text.split('\n'):
                    if line.strip():
                        f.write(f"\"{line.strip()}\"\n")

            print("Ollama analysis saved to 'ollama_analyse.csv'.")

        except Exception as e:
            print(f"Error running Ollama analysis: {e}")




        context = {"columns"  : column_names ,
                   'csv_file': file_name,
                   'save_path':save_path,
                #    'ollama_result' : ollama_text,
                   
                   }

        return render(request, 'dashboard/analysis_dashboard.html', context)
    # return render(request, 'dashboard/analysis_dashboard.html')


import chardet
def upload_csv(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        uploaded_file = request.FILES['csv_file']
        file_name = uploaded_file.name

        # Save file to local project directory
        save_path = os.path.join(os.path.dirname(__file__), file_name)
        with open(save_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        try:
            # Use ISO-8859-1 or latin1 to safely decode odd characters like 0x81
            df = pd.read_csv(save_path, encoding='ISO-8859-1', on_bad_lines='skip')
        except Exception as e:
            return HttpResponse(f"Could not decode file. Error: {str(e)}", status=400)




        # Optional: Load the file using pandas to confirm it's accessible
        # df = pd.read_csv(save_path, encoding='cp1252')
        df = pd.read_csv(save_path)

        print("fjosdifdhjoisfjhdojhsoifdjoi---------------------------------------------------------------")
        print("Saving file to:", save_path)
        print("Uploaded file name:", file_name) 

        print(df.head())  # For debug

        column_names = df.columns.tolist()
        context = {"columns"  : column_names ,
                   'csv_file': file_name,
                   
                   }

        # return HttpResponse(f"File '{file_name}' uploaded and saved to {save_path}.")

    return render(request, 'main_page/upload.html' , context)


# #########################################################################################################################
###########################################################################################################################


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from django.urls import reverse
from django.contrib import messages
import json

@csrf_exempt
def train_model(request):
    if request.method == 'POST':
        x_targets = request.POST.getlist('x_targets')  # List of selected X columns
        y_target = request.POST.get('y_target')        # Single Y column
        csv_file = request.POST.get('csv_file')        # CSV file name sent as hidden field

        # Load the CSV again
        csv_path = os.path.join(os.path.dirname(__file__), csv_file)
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='cp1252')

        # Debug: print selections
        print("X Targets:", x_targets)
        print("Y Target:", y_target)

#  now we work on AI ###########################################
        # Only keep relevant columns (X + y)
        selected_columns = x_targets + [y_target]
        df = df[selected_columns]

        # Handle missing values and encode non-numeric columns
        label_encoders = {}

        for col in df.columns:
            if df[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(df[col]):
                le = LabelEncoder()
                try:
                    df[col] = df[col].astype(str).fillna("NA")
                    df[col] = le.fit_transform(df[col])
                    label_encoders[col] = le
                except:
                    pass
            else:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].mean())

                # Convert float to int
                if pd.api.types.is_float_dtype(df[col]):
                    df[col] = df[col].astype(int)

        # Separate features and label
        X = df[x_targets]
        y = df[y_target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature scaling
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


#  now we work on AI ###################################################################################
#  now we work on AI ###################################################################################
        # Models to compare
        models = {
            'Random Forest': RandomForestRegressor(random_state=42),
            'Bagging Regressor': BaggingRegressor(estimator=DecisionTreeRegressor(), random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Extra Trees Regressor': ExtraTreesRegressor(random_state=42),
            'XGB Regressor': XGBRegressor(random_state=42),
            'Hist Gradient Boosting': HistGradientBoostingRegressor(random_state=42)
        }

        results = []
        feature_importances = {}
        predictions = {}

        # Train and evaluate models
        for name, model in models.items():
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time

            y_pred = model.predict(X_test_scaled)
            predictions[name] = y_pred

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results.append({
                'Model': name,
                'Training Time (s)': training_time,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R² Score': r2
            })

            if hasattr(model, 'feature_importances_'):
                feature_importances[name] = dict(zip(X.columns, model.feature_importances_))

        # Sort by R² Score
        results_df = pd.DataFrame(results).sort_values('R² Score', ascending=False).reset_index(drop=True)


        # Save metrics to CSV

        # Output: Model Performance Summary
        print("\n=== Model Performance Summary ===")
        for _, row in results_df.iterrows():
            print(f"{row['Model']}: Time={row['Training Time (s)']:.4f}s, "
                f"R²={row['R² Score']:.4f}, MSE={row['MSE']:.4f}, "
                f"RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}")
            
        results_df.to_csv('C:/Users/adity/projects/dataset_analysis/static/csv/metrics.csv', index=False)
        # results_df.to_csv('C:/Users/adity/projects/dataset_analysis/templates/Dashboard/metrics.csv', index=False)
        print("✓ Metrics saved to 'metrics.csv'")


        # Output: Feature Importances
        feature_importance_data = []
        print("\n=== Feature Importances ===")
        for model_name, importances in feature_importances.items():
            print(f"\n{model_name}:")
            for feature, value in importances.items():
                feature_importance_data.append({
                            'Model': model_name,
                            'Feature': feature,
                            'Importance': value
                        })
                print(f"  {feature}: {value:.4f}")

        if feature_importance_data:
            feature_df = pd.DataFrame(feature_importance_data)
            feature_df.to_csv('C:/Users/adity/projects/dataset_analysis/static/csv/Feature_Importances.csv', index=False)
            # feature_df.to_csv('C:/Users/adity/projects/dataset_analysis/templates/Dashboard/Feature_Importances.csv', index=False)
            print("✓ Feature importances saved to 'Feature_Importances.csv'")
                    

        # Output: Actual vs Predicted (First 100)
        actual_vs_pred_data = []
        print("\n=== Actual vs Predicted (First 100 values) ===")
        for model_name, y_pred in predictions.items():
            print(f"\n{model_name}:")
            for i, (actual, predicted) in enumerate(zip(y_test[:500], y_pred[:500])):
                actual_vs_pred_data.append({
                            'Model': model_name,
                            'Index': i,
                            'Actual': actual,
                            'Predicted': predicted
                        })
                print(f"  Actual: {actual:.2f}, Predicted: {predicted:.2f}")


        actual_pred_df = pd.DataFrame(actual_vs_pred_data)
        actual_pred_df.to_csv('C:/Users/adity/projects/dataset_analysis/static/csv/Actual_vs_Predicted.csv', index=False)
        # actual_pred_df.to_csv('C:/Users/adity/projects/dataset_analysis/templates/Dashboard/Actual_vs_Predicted.csv', index=False)
        print("✓ Actual vs Predicted data saved to 'Actual_vs_Predicted.csv'")

        # Recommend Best Model
        # def recommend_best_model(df):
        # Recommend Best Model
        # Recommend Best Model
        best = results_df.iloc[0]
        reasons = []

        if best['R² Score'] >= 0.9:
            reasons.append("excellent accuracy (R² ≥ 0.9)")
        if best['MAE'] == results_df['MAE'].min():
            reasons.append("lowest Mean Absolute Error (MAE)")
        if best['MSE'] == results_df['MSE'].min():
            reasons.append("lowest Mean Squared Error (MSE)")
        if best['Training Time (s)'] <= results_df['Training Time (s)'].median():
            reasons.append("fast training time")

        best_model = (f"\n🚀 Recommended Best Model: {best['Model']}")
        why_best = ("Why it's recommended:", ", ".join(reasons) + ".")
        print(best_model)
        print(why_best)

        print("\n📁 All CSV files have been generated successfully!")
        print("Files created:")
        print("- metrics.csv")
        print("- Feature_Importances.csv") 
        print("- Actual_vs_Predicted.csv")

        # recommend_best_model(results_df)        

#  now we end work on AI ###################################################################################
#  now we start ollama on AI ###################################################################################
        try:
            # Path to your CSV data analysis folder
            analysis_file = "C:/Users/adity/projects/dataset_analysis/static/csv"

            # Collect all CSV content
            data_summary = {}
            for csv_file in [ "metrics.csv" ]:
                path = os.path.join(analysis_file, csv_file)
                if os.path.exists(path):
                    df_temp = pd.read_csv(path)
                    data_summary[csv_file] = df_temp.to_dict(orient='records')
                else:
                    data_summary[csv_file] = "File not found."

            # Convert collected data into prompt
            ollama_prompt = f"""
            You are a data scientist. Here are CSV summaries of a trained machine learning model on dataset.
            this ml models inclue Random Forest ,Bagging Regressor ,Decision Tree ,Extra Trees Regressor ,XGB Regressor ,Hist Gradient Boosting 
            and tell whcih is best.
            Format your response as a structured summary

            Data Summary:
            {json.dumps(data_summary, indent=2)}
            """
            pro = ""
            prompt = f"{best_model}\n{why_best}"

            # Call local Ollama API
            response = requests.post(
                            "http://localhost:11434/api/generate",
                            json={
                                # "model": "tinydolphin",
                                 "model": "mistral",
                                "prompt":  prompt  ,
                                # "prompt": "this ml models inclue Random Forest ,Bagging Regressor ,Decision Tree ,Extra Trees Regressor ,XGB Regressor ,Hist Gradient Boosting and tell whcih is best and why.",
                                "stream": False
                            }
                        )


            result = response.json()
            ollama_text = result.get("response", "Ollama returned no response.")

            # Save Ollama analysis
            with open(os.path.join(analysis_file, "ml_ollama_analyse.csv"), "w", encoding="utf-8") as f:
                f.write("Insight\n")
                for line in ollama_text.split('\n'):
                    if line.strip():
                        f.write(f"\"{line.strip()}\"\n")

            print("Ollama analysis saved to 'ml_ollama_analyse.csv'.")

        except Exception as e:
            print(f"Error running Ollama analysis: {e}")



    # Store results as JSON in session to access in dashboard
    request.session['model_results'] = results_df.to_dict(orient='records')

    # Optionally print to debug
    print("📊 Data passed to dashboard:", request.session['model_results'])

    return redirect('dashboard')

    #     context = {
    #          'x_targets': x_targets,
    #         'y_target': y_target,
    #         'data_preview': df.head().to_html(),
    #         'train_shape': X_train.shape,
    #         'test_shape': X_test.shape,

    #     }

    #     return render(request, 'main_page/train_model.html', context)

    # return HttpResponse("Invalid request method.", status=405)



# def dashboard(request):
#     model_results = request.session.get('model_results', [])
#     return render(request, 'dashboard/dashboard.html', {'model_results': model_results})

def dashboard(request):
    return render(request, 'Dashboard/dashboard.html')
