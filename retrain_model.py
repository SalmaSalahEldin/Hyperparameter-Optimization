#!/usr/bin/env python3
"""
Retrain the Student Performance Model with current scikit-learn version
This fixes the OneHotEncoder compatibility issue
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

def create_directories():
    """Create separate directories for baseline and optimized models"""
    print("Creating artifact directories...")
    
    # Remove existing artifacts directory
    if os.path.exists('artifacts'):
        import shutil
        shutil.rmtree('artifacts')
        print("Removed existing artifacts directory")
    
    # Create new directory structure
    os.makedirs('artifacts/baseline_models', exist_ok=True)
    os.makedirs('artifacts/optimized_models', exist_ok=True)
    os.makedirs('artifacts/baseline_preprocessors', exist_ok=True)
    os.makedirs('artifacts/optimized_preprocessors', exist_ok=True)
    os.makedirs('artifacts/metrics', exist_ok=True)
    os.makedirs('artifacts/training_logs', exist_ok=True)
    
    print("Created directory structure:")
    print("  - artifacts/baseline_models/")
    print("  - artifacts/optimized_models/")
    print("  - artifacts/baseline_preprocessors/")
    print("  - artifacts/optimized_preprocessors/")
    print("  - artifacts/metrics/")
    print("  - artifacts/training_logs/")

def save_metrics(metrics, filename):
    """Save metrics to JSON file"""
    metrics_path = f'artifacts/metrics/{filename}'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to: {metrics_path}")

def save_training_log(log_data, filename):
    """Save training log to JSON file"""
    log_path = f'artifacts/training_logs/{filename}'
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    print(f"Training log saved to: {log_path}")

def train_baseline_models(X_train, y_train, X_test, y_test):
    """Train models without hyperparameter optimization"""
    print("\n" + "="*60)
    print("TRAINING BASELINE MODELS (No Optimization)")
    print("="*60)
    
    baseline_models = {
        'Linear_Regression': LinearRegression(),
        'Ridge_Regression': Ridge(random_state=42),
        'Lasso_Regression': Lasso(random_state=42),
        'Random_Forest': RandomForestRegressor(random_state=42),
        'KNeighbors': KNeighborsRegressor(),
        'Decision_Tree': DecisionTreeRegressor(random_state=42)
    }
    
    baseline_results = {}
    baseline_metrics = {
        'training_timestamp': datetime.now().isoformat(),
        'model_type': 'baseline',
        'models': {}
    }
    
    for name, model in baseline_models.items():
        print(f"\nTraining {name}...")
        
        # Train model with default parameters
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate comprehensive metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Store results
        baseline_results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
        
        # Store metrics for saving
        baseline_metrics['models'][name] = {
            'parameters': 'default',
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'overfitting_score': float(train_r2 - test_r2)
        }
        
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Training RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Training MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Overfitting Score: {train_r2 - test_r2:.4f}")
        
        # Save baseline model
        model_path = f'artifacts/baseline_models/{name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  Saved to: {model_path}")
    
    # Save baseline metrics
    save_metrics(baseline_metrics, 'baseline_metrics.json')
    
    return baseline_results

def train_optimized_models(X_train, y_train, X_test, y_test):
    """Train models with hyperparameter optimization"""
    print("\n" + "="*60)
    print("TRAINING OPTIMIZED MODELS (With Hyperparameter Tuning)")
    print("="*60)
    
    # Define models and their parameter grids for optimization
    optimization_configs = {
        'Ridge_Regression': {
            'model': Ridge(random_state=42),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky']
            }
        },
        'Lasso_Regression': {
            'model': Lasso(random_state=42),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'max_iter': [1000, 2000]
            }
        },
        'Random_Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'KNeighbors': {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'Decision_Tree': {
            'model': DecisionTreeRegressor(random_state=42),
            'params': {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    }
    
    optimized_results = {}
    best_overall_score = -float('inf')
    best_overall_model = None
    best_overall_name = None
    
    optimized_metrics = {
        'training_timestamp': datetime.now().isoformat(),
        'model_type': 'optimized',
        'models': {}
    }
    
    for name, config in optimization_configs.items():
        print(f"\nOptimizing {name}...")
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            config['model'], 
            config['params'], 
            cv=5, 
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")
        
        # Make predictions with optimized model
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        # Calculate comprehensive metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Store results
        optimized_results[name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
        
        # Store metrics for saving
        optimized_metrics['models'][name] = {
            'best_parameters': grid_search.best_params_,
            'cv_score': float(grid_search.best_score_),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'overfitting_score': float(train_r2 - test_r2)
        }
        
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Training RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Training MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Overfitting Score: {train_r2 - test_r2:.4f}")
        
        # Save optimized model
        model_path = f'artifacts/optimized_models/{name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"  Saved to: {model_path}")
        
        # Track best overall model
        if grid_search.best_score_ > best_overall_score:
            best_overall_score = grid_search.best_score_
            best_overall_model = best_model
            best_overall_name = name
    
    # Save optimized metrics
    save_metrics(optimized_metrics, 'optimized_metrics.json')
    
    print(f"\nBest Overall Optimized Model: {best_overall_name}")
    print(f"   Cross-validation score: {best_overall_score:.4f}")
    
    return optimized_results, best_overall_model, best_overall_name

def save_preprocessors(preprocessor, is_optimized=False):
    """Save preprocessors in appropriate directory"""
    if is_optimized:
        path = 'artifacts/optimized_preprocessors/preprocessor.pkl'
    else:
        path = 'artifacts/baseline_preprocessors/preprocessor.pkl'
    
    with open(path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print(f"Preprocessor saved to: {path}")

def compare_models(baseline_results, optimized_results):
    """Compare baseline vs optimized model performance"""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON: BASELINE vs OPTIMIZED")
    print("="*80)
    
    print(f"{'Model':<25} {'Baseline Test R²':<18} {'Optimized Test R²':<18} {'Improvement':<12}")
    print("-" * 80)
    
    comparison_data = {
        'comparison_timestamp': datetime.now().isoformat(),
        'models': {}
    }
    
    for name in baseline_results.keys():
        if name in optimized_results:
            baseline_r2 = baseline_results[name]['test_r2']
            optimized_r2 = optimized_results[name]['test_r2']
            improvement = optimized_r2 - baseline_r2
            
            comparison_data['models'][name] = {
                'baseline_test_r2': float(baseline_r2),
                'optimized_test_r2': float(optimized_r2),
                'improvement': float(improvement),
                'improvement_percentage': float((improvement / baseline_r2) * 100) if baseline_r2 != 0 else 0.0
            }
            
            print(f"{name:<25} {baseline_r2:<18.4f} {optimized_r2:<18.4f} {improvement:<+12.4f}")
    
    # Save comparison metrics
    save_metrics(comparison_data, 'model_comparison.json')
    
    print("\n" + "="*80)

def generate_summary_report(baseline_results, optimized_results, best_model_name):
    """Generate a comprehensive summary report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE TRAINING SUMMARY REPORT")
    print("="*80)
    
    summary = {
        'training_summary': {
            'timestamp': datetime.now().isoformat(),
            'total_baseline_models': len(baseline_results),
            'total_optimized_models': len(optimized_results),
            'best_model': best_model_name,
            'dataset_info': {
                'total_samples': 1000,
                'train_samples': 800,
                'test_samples': 200,
                'features': 14
            }
        },
        'performance_highlights': {
            'best_baseline_model': max(baseline_results.items(), key=lambda x: x[1]['test_r2'])[0],
            'best_optimized_model': best_model_name,
            'most_improved_model': None,
            'biggest_improvement': 0.0
        },
        'recommendations': []
    }
    
    # Find most improved model
    max_improvement = 0
    most_improved = None
    
    for name in baseline_results.keys():
        if name in optimized_results:
            improvement = optimized_results[name]['test_r2'] - baseline_results[name]['test_r2']
            if improvement > max_improvement:
                max_improvement = improvement
                most_improved = name
    
    summary['performance_highlights']['most_improved_model'] = most_improved
    summary['performance_highlights']['biggest_improvement'] = float(max_improvement)
    
    # Generate recommendations
    if max_improvement > 0.05:
        summary['recommendations'].append(f"Use optimized {most_improved} for significant performance gain (+{max_improvement:.3f})")
    
    if optimized_results[best_model_name]['test_r2'] > 0.85:
        summary['recommendations'].append(f"Excellent performance achieved with {best_model_name} (R² > 0.85)")
    
    summary['recommendations'].append("Always use cross-validation for hyperparameter tuning")
    summary['recommendations'].append("Monitor overfitting scores to ensure model generalization")
    
    # Save summary report
    save_metrics(summary, 'training_summary.json')
    
    # Print summary
    print(f"Total Models Trained: {len(baseline_results)} baseline + {len(optimized_results)} optimized")
    print(f"Best Overall Model: {best_model_name}")
    print(f"Most Improved Model: {most_improved} (+{max_improvement:.4f})")
    print(f"Dataset: {summary['training_summary']['dataset_info']['total_samples']} samples, {summary['training_summary']['dataset_info']['features']} features")
    print("\nRecommendations:")
    for rec in summary['recommendations']:
        print(f"  • {rec}")
    
    print("\n" + "="*80)

def main():
    """Main training function"""
    print("Starting comprehensive model training...")
    print("This will create both baseline and optimized models")
    
    # Create directory structure
    create_directories()
    
    # Load the data
    print("\nLoading data...")
    df = pd.read_csv('notebook/data/stud.csv')
    print(f"Data shape: {df.shape}")
    
    # Prepare X and y
    X = df.drop(columns=['math_score'], axis=1)
    y = df['math_score']
    
    # Identify categorical and numerical features
    cat_features = X.select_dtypes(include="object").columns
    num_features = X.select_dtypes(exclude="object").columns
    
    print(f"Categorical features: {cat_features.tolist()}")
    print(f"Numerical features: {num_features.tolist()}")
    
    # Create preprocessor
    print("\nCreating preprocessor...")
    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("OneHotEncoder", oh_transformer, cat_features),
            ("StandardScaler", numeric_transformer, num_features),
        ],
        remainder='passthrough'
    )
    
    # Transform the data
    print("Transforming data...")
    X_transformed = preprocessor.fit_transform(X)
    print(f"Transformed X shape: {X_transformed.shape}")
    
    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42
    )
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Save baseline preprocessor
    save_preprocessors(preprocessor, is_optimized=False)
    
    # Train baseline models (no optimization)
    baseline_results = train_baseline_models(X_train, y_train, X_test, y_test)
    
    # Train optimized models (with hyperparameter tuning)
    optimized_results, best_model, best_name = train_optimized_models(X_train, y_train, X_test, y_test)
    
    # Save optimized preprocessor
    save_preprocessors(preprocessor, is_optimized=True)
    
    # Compare model performance
    compare_models(baseline_results, optimized_results)
    
    # Generate comprehensive summary
    generate_summary_report(baseline_results, optimized_results, best_name)
    
    # Save best model separately for easy access
    best_model_path = 'artifacts/best_model.pkl'
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"\nBest overall model saved to: {best_model_path}")
    print("\nTraining completed successfully!")
    print("\nDirectory structure created:")
    print("  artifacts/")
    print("  ├── baseline_models/          (models without optimization)")
    print("  ├── optimized_models/         (models with hyperparameter tuning)")
    print("  ├── baseline_preprocessors/   (preprocessors for baseline models)")
    print("  ├── optimized_preprocessors/  (preprocessors for optimized models)")
    print("  ├── metrics/                  (detailed performance metrics)")
    print("  ├── training_logs/            (training logs and summaries)")
    print("  └── best_model.pkl            (best performing optimized model)")
    print("\nMetrics and logs saved for analysis and documentation!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc() 