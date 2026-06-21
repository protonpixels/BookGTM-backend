from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
from typing import Optional
from datetime import datetime, timedelta

app = FastAPI(title="Book Downloads Predictor", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "https://bookgtm.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and artifacts
print("Loading model and artifacts...")
model = joblib.load('models/book_download_predictor.joblib')
feature_names = joblib.load('models/feature_names.joblib')
available_categories = joblib.load('models/available_categories.joblib')
month_mapping = joblib.load('models/month_mapping.joblib')
model_metadata = joblib.load('models/model_metadata.joblib')

print(f"✅ Model loaded successfully!")
print(f"📊 Available categories: {len(available_categories)}")
print(f"🔢 Features expected: {len(feature_names)}")

# Months list
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
global_median = model_metadata.get('global_median', 1000)


# ============================================
# Request Models - NO PUBLISHER NAME REQUIRED!
# ============================================

class BookPredictionRequest(BaseModel):
    title: str = Field(..., example="The Art of Programming")
    category: str = Field(..., example="Self-Improvement")
    pages: int = Field(..., example=350)
    days_since_published: int = Field(..., example=30)
    published_month: Optional[str] = Field(None, example="Jan")
    publisher_book_count: Optional[int] = Field(3, example=3,
                                                description="How many books has this publisher already published? (1-100)")

    class Config:
        schema_extra = {
            "example": {
                "title": "The Art of Programming",
                "category": "Self-Improvement",
                "pages": 350,
                "days_since_published": 30,
                "published_month": "Jan",
                "publisher_book_count": 3
            }
        }


class PredictionResponse(BaseModel):
    predicted_downloads: float
    log_transformed_prediction: float
    confidence_interval: Optional[dict] = None
    feature_contributions: Optional[dict] = None
    input_summary: dict


# ============================================
# Helper Functions
# ============================================

def get_publisher_features(book_count):
    """Calculate publisher features based on book count"""
    # Use the smoothed median based on book count
    # More books = higher median downloads (generally)
    if book_count <= 0:
        book_count = 1

    # Base median increases with more books
    # This is a heuristic - adjust based on your data
    base_median = global_median

    # Scale: 1 book = 0.3x, 5 books = 0.6x, 20+ books = 1.0x
    scale = min(1.0, 0.3 + (book_count / 20) * 0.7)
    publisher_median = base_median * scale

    # Add some randomness reduction for larger publishers
    # More books = more stable predictions
    publisher_median_smoothed = publisher_median

    return {
        'publisher_median_smoothed': publisher_median_smoothed,
        'publisher_book_count': book_count
    }


def prepare_features(request: BookPredictionRequest):
    """Prepare feature vector for prediction"""

    # Calculate published date
    dataset_creation_date = datetime(2026, 4, 1)
    published_date = dataset_creation_date - timedelta(days=request.days_since_published)

    # Use provided month or calculate from date
    published_month = request.published_month or months[published_date.month - 1]
    published_year = published_date.year

    # Title features
    title_word_count = len(request.title.split())
    title_length = len(request.title)
    title_has_number = int(any(char.isdigit() for char in request.title))
    title_has_guide = int('guide' in request.title.lower())
    title_has_handbook = int('handbook' in request.title.lower())

    # Publisher features based on book count
    publisher_features = get_publisher_features(request.publisher_book_count or 3)

    # Pages
    pages = request.pages
    pages_log = np.log1p(pages)

    # Month and quarter
    month_num = month_mapping[published_month]
    quarter = (month_num - 1) // 3 + 1

    # Category dummy (one-hot encoded)
    category_dummies = {}
    for cat in available_categories:
        col_name = f'Category_{cat}'
        category_dummies[col_name] = 1.0 if cat == request.category else 0.0

    # Month dummies
    month_dummies = {}
    for i in range(1, 13):
        col_name = f'Month_{i}'
        month_dummies[col_name] = 1.0 if i == month_num else 0.0

    # Quarter dummies
    quarter_dummies = {}
    for i in range(1, 5):
        col_name = f'Quarter_{i}'
        quarter_dummies[col_name] = 1.0 if i == quarter else 0.0

    # Days since published squared
    days_since_published_squared = request.days_since_published ** 2

    # Combine all features in the correct order
    feature_dict = {
        **category_dummies,
        **month_dummies,
        **quarter_dummies,
        'title_word_count': title_word_count,
        'title_length': title_length,
        'days_since_published': request.days_since_published,
        'days_since_published_squared': days_since_published_squared,
        'pages': pages,
        'pages_log': pages_log,
        'publisher_median_smoothed': publisher_features['publisher_median_smoothed'],
        'publisher_book_count': publisher_features['publisher_book_count']
    }

    # Ensure all features are present
    missing_features = []
    for feature in feature_names:
        if feature not in feature_dict:
            missing_features.append(feature)

    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Create DataFrame with correct column order
    X = pd.DataFrame([feature_dict])[feature_names]

    return X, feature_dict


# ============================================
# API Endpoints
# ============================================

@app.get("/")
def read_root():
    return {
        "message": "Book Downloads Predictor API",
        "version": "1.0",
        "status": "online"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_count": len(feature_names)
    }


@app.get("/categories")
def get_categories():
    """Get list of available categories"""
    return {"categories": available_categories}


@app.get("/months")
def get_months():
    """Get list of available months"""
    return {"months": months}


@app.post("/predict", response_model=PredictionResponse)
def predict_downloads(request: BookPredictionRequest):
    """Predict downloads for a book"""

    try:
        # Prepare features
        X, feature_dict = prepare_features(request)

        # Make prediction (log scale)
        prediction_log = model.predict(X)[0]

        # Convert back to original scale
        prediction = np.expm1(prediction_log)

        # Get normalized feature importance
        feature_importance = model.get_feature_importance()
        total_importance = sum(feature_importance)
        normalized_importance = [(imp / total_importance * 100) for imp in feature_importance]

        # Get top 5 features
        importance_pairs = list(zip(feature_names, normalized_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        top_features = importance_pairs[:5]

        return PredictionResponse(
            predicted_downloads=round(prediction, 2),
            log_transformed_prediction=round(prediction_log, 4),
            confidence_interval={
                "lower": round(prediction * 0.8, 2),
                "upper": round(prediction * 1.2, 2)
            },
            feature_contributions={
                "top_features": [
                    {"feature": name, "importance": round(imp, 2)}
                    for name, imp in top_features
                ]
            },
            input_summary={
                "title": request.title,
                "category": request.category,
                "pages": request.pages,
                "days_since_published": request.days_since_published,
                "published_month": request.published_month or "Auto-detected",
                "publisher_book_count": request.publisher_book_count or 3
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)