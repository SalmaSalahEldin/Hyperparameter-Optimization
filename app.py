from fastapi import FastAPI
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np
import pandas as pd
import uvicorn

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Enums for API field options
class Gender(str, Enum):
    male = "male"
    female = "female"

class Ethnicity(str, Enum):
    group_a = "group A"
    group_b = "group B"
    group_c = "group C"
    group_d = "group D"
    group_e = "group E"

class ParentalEducation(str, Enum):
    some_high_school = "some high school"
    high_school = "high school"
    some_college = "some college"
    associates_degree = "associate's degree"
    bachelors_degree = "bachelor's degree"
    masters_degree = "master's degree"

class LunchType(str, Enum):
    free_reduced = "free/reduced"
    standard = "standard"

class TestPrepCourse(str, Enum):
    none = "none"
    completed = "completed"

# Pydantic model for API requests
class PredictionRequest(BaseModel):
    gender: Gender
    ethnicity: Ethnicity
    parental_level_of_education: ParentalEducation
    lunch: LunchType
    test_preparation_course: TestPrepCourse
    reading_score: float = Field(default=75.0, ge=0, le=100, description="Reading score from 0-100 (default: 75)")
    writing_score: float = Field(default=66.0, ge=0, le=100, description="Writing score from 0-100 (default: 66)")

    class Config:
        schema_extra = {
            "example": {
                "gender": "female",
                "ethnicity": "group A",
                "parental_level_of_education": "bachelor's degree",
                "lunch": "standard",
                "test_preparation_course": "completed",
                "reading_score": 75.0,
                "writing_score": 66.0
            }
        }

# Pydantic model for API responses
class PredictionResponse(BaseModel):
    prediction: float
    input_data: dict
    status: str
    message: str

# Create FastAPI app
app = FastAPI(
    title="Academic Performance Analytics API",
    description="""
    ## Academic Performance Analytics API
    
    This API provides machine learning-powered predictions for student academic performance in mathematics based on comprehensive demographic and educational factors.
    
    ### Available Field Values:
    
    **Gender**: male, female
    **Ethnicity**: group A, group B, group C, group D, group E
    **Parental Education**: some high school, high school, some college, associate's degree, bachelor's degree, master's degree
    **Lunch**: free/reduced, standard
    **Test Prep Course**: none, completed
    **Reading Score**: 0-100
    **Writing Score**: 0-100
    """,
    version="1.0.0"
)

@app.post("/predict", response_model=PredictionResponse)
async def predict_api(request: PredictionRequest):
    """
    Make a prediction for student math performance.
    
    Returns the predicted math score based on input parameters.
    """
    try:
        # Create CustomData object
        data = CustomData(
            gender=request.gender,
            race_ethnicity=request.ethnicity,
            parental_level_of_education=request.parental_level_of_education,
            lunch=request.lunch,
            test_preparation_course=request.test_preparation_course,
            reading_score=request.reading_score,
            writing_score=request.writing_score
        )
        
        # Get data as dataframe
        pred_df = data.get_data_as_data_frame()
        print(f"Input data: {pred_df}")
        
        # Make prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Apply bounds to prediction (0-100)
        raw_prediction = float(results[0])
        bounded_prediction = max(0.0, min(100.0, raw_prediction))
        
        # Log if bounds were applied
        if raw_prediction != bounded_prediction:
            print(f"Warning: Prediction {raw_prediction:.2f} was bounded to {bounded_prediction:.2f}")
        
        # Return JSON response
        return PredictionResponse(
            prediction=bounded_prediction,
            input_data=request.dict(),
            status="success",
            message="Prediction completed successfully"
        )
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return PredictionResponse(
            prediction=0.0,
            input_data=request.dict(),
            status="error",
            message=f"Prediction failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Academic Performance Analytics API is running"}

@app.get("/options")
async def get_form_options():
    """
    Get available options for all form fields.
    
    Useful for building dynamic forms or validating inputs.
    """
    return {
        "gender": [gender.value for gender in Gender],
        "ethnicity": [ethnicity.value for ethnicity in Ethnicity],
        "parental_level_of_education": [edu.value for edu in ParentalEducation],
        "lunch": [lunch.value for lunch in LunchType],
        "test_preparation_course": [course.value for course in TestPrepCourse]
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5001,
        reload=True,
        log_level="info"
    )        


