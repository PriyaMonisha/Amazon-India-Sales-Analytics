from feast import FeatureService

from feature_views import customer_training_fv  # NO feast_repo. prefix

churn_feature_service = FeatureService(
    name="churn_features",
    features=[customer_training_fv],
)
