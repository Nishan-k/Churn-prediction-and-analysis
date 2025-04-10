import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


# Load the model:
model = joblib.load("customer_churn_ml/churn_clf_model.pkl")

def aggregated_shap_features(customer_data, model=model, background_data=None):
    """
    Creates a clean two-row SHAP dashboard:
    - Top row: SHAP impact chart (without values on bars)
    - Bottom row: Feature values table
    - Perfect for Jupyter notebooks with no overlapping
    """
    

    # ====================== [1. SHAP CALCULATIONS] ======================
    # Extract model components
    final_model = model.named_steps['model']
    preprocessor = model.named_steps['preprocessor']
    
    # Transform customer data:
    X_transformed = preprocessor.transform(customer_data)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    
    # Get feature names
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
    
    # Create explainer
    if hasattr(final_model, 'predict_proba'):
        if hasattr(final_model, 'estimators_'):
            explainer = shap.TreeExplainer(final_model)
        elif hasattr(final_model, 'coef_'):
            explainer = shap.LinearExplainer(final_model, X_transformed)
        else:
            if background_data is not None:
                X_background = preprocessor.transform(background_data)
                if hasattr(X_background, "toarray"):
                    X_background = X_background.toarray()
                explainer = shap.KernelExplainer(final_model.predict_proba, X_background)
            else:
                explainer = shap.Explainer(final_model, X_transformed, feature_names=feature_names)
    else:
        explainer = shap.Explainer(final_model, X_transformed, feature_names=feature_names)
    
    # Get SHAP values
    if isinstance(explainer, shap.KernelExplainer):
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_for_churn = shap_values[1] 
        else:
            shap_for_churn = shap_values
        base_value = explainer.expected_value
        if isinstance(base_value, list) and len(base_value) > 1:
            base_value = base_value[1]
    else:
        shap_values = explainer(X_transformed)
        if len(shap_values.shape) > 2 and shap_values.shape[2] > 1:
            shap_for_churn = shap_values[:, :, 1].values
            base_value = shap_values[0, :, 1].base_values
        else:
            shap_for_churn = shap_values.values
            base_value = shap_values.base_values
    
    # Feature mapping
    original_features = customer_data.columns.tolist()
    
    feature_mapping = {}
    for i, feature_name in enumerate(feature_names):
        parts = feature_name.split('__')
        if feature_name.startswith('encoding__') and len(parts) >= 2:
            full_feature = parts[1]
            for orig_feature in original_features:
                if full_feature.startswith(orig_feature):
                    feature_mapping[i] = orig_feature
                    break
            else:
                feature_mapping[i] = '_'.join(full_feature.split('_')[:-1])
        elif feature_name.startswith('remainder__') and len(parts) >= 2:
            feature_mapping[i] = parts[1]
        else:
            feature_mapping[i] = feature_name
    
    # Aggregate SHAP values
    aggregated_shap = {}
    for i, shap_value in enumerate(shap_for_churn[0]):
        original_feature = feature_mapping.get(i)
        if original_feature not in aggregated_shap:
            aggregated_shap[original_feature] = 0
        aggregated_shap[original_feature] += shap_value
    
    # Get customer values
    customer_values = {}
    for feature in original_features:
        if feature in customer_data.columns:
            customer_values[feature] = str(customer_data.iloc[0][feature])
    
   
    sorted_dict = dict(sorted(aggregated_shap.items(), key=lambda item: item[1], reverse=True))
    return sorted_dict









def create_clean_shap_dashboard(customer_data, model=model, background_data=None):
    """
    Creates a clean two-row SHAP dashboard:
    - Top row: SHAP impact chart (without values on bars)
    - Bottom row: Feature values table
    - Perfect for Jupyter notebooks with no overlapping
    """
    

    # ====================== [1. SHAP CALCULATIONS] ======================
    # Extract model components
    final_model = model.named_steps['model']
    preprocessor = model.named_steps['preprocessor']
    
    # Transform customer data
    X_transformed = preprocessor.transform(customer_data)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    
    # Get feature names
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
    
    # Create explainer
    if hasattr(final_model, 'predict_proba'):
        if hasattr(final_model, 'estimators_'):
            explainer = shap.TreeExplainer(final_model)
        elif hasattr(final_model, 'coef_'):
            explainer = shap.LinearExplainer(final_model, X_transformed)
        else:
            if background_data is not None:
                X_background = preprocessor.transform(background_data)
                if hasattr(X_background, "toarray"):
                    X_background = X_background.toarray()
                explainer = shap.KernelExplainer(final_model.predict_proba, X_background)
            else:
                explainer = shap.Explainer(final_model, X_transformed, feature_names=feature_names)
    else:
        explainer = shap.Explainer(final_model, X_transformed, feature_names=feature_names)
    
    # Get SHAP values
    if isinstance(explainer, shap.KernelExplainer):
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_for_churn = shap_values[1] 
        else:
            shap_for_churn = shap_values
        base_value = explainer.expected_value
        if isinstance(base_value, list) and len(base_value) > 1:
            base_value = base_value[1]
    else:
        shap_values = explainer(X_transformed)
        if len(shap_values.shape) > 2 and shap_values.shape[2] > 1:
            shap_for_churn = shap_values[:, :, 1].values
            base_value = shap_values[0, :, 1].base_values
        else:
            shap_for_churn = shap_values.values
            base_value = shap_values.base_values
    
    # Feature mapping
    original_features = customer_data.columns.tolist()
    
    feature_mapping = {}
    for i, feature_name in enumerate(feature_names):
        parts = feature_name.split('__')
        if feature_name.startswith('encoding__') and len(parts) >= 2:
            full_feature = parts[1]
            for orig_feature in original_features:
                if full_feature.startswith(orig_feature):
                    feature_mapping[i] = orig_feature
                    break
            else:
                feature_mapping[i] = '_'.join(full_feature.split('_')[:-1])
        elif feature_name.startswith('remainder__') and len(parts) >= 2:
            feature_mapping[i] = parts[1]
        else:
            feature_mapping[i] = feature_name
    
    # Aggregate SHAP values
    aggregated_shap = {}
    for i, shap_value in enumerate(shap_for_churn[0]):
        original_feature = feature_mapping.get(i)
        if original_feature not in aggregated_shap:
            aggregated_shap[original_feature] = 0
        aggregated_shap[original_feature] += shap_value
    
    # Get customer values
    customer_values = {}
    for feature in original_features:
        if feature in customer_data.columns:
            customer_values[feature] = str(customer_data.iloc[0][feature])
    
    # Sort features by absolute impact
    sorted_features = sorted(aggregated_shap.keys(), 
                           key=lambda x: abs(aggregated_shap[x]), 
                           reverse=True)
    sorted_values = [aggregated_shap[f] for f in sorted_features]
    
    
    # ====================== [2. VISUALIZATION] ======================
    # Create figure with two rows
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.4)
    
    # ===== TOP ROW: CLEAN SHAP CHART (NO VALUES ON BARS) =====
    ax1 = fig.add_subplot(gs[0])
    
    # Visual parameters
    bar_height = 0.7
    y_pos = np.arange(len(sorted_features))
    colors = ['#FF4560' if x < 0 else '#008FFB' for x in sorted_values]
    
    # Create clean bars without values
    ax1.barh(y_pos, sorted_values, height=bar_height, color=colors)
    
    # Chart styling
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sorted_features, fontsize=12)
    ax1.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
    ax1.axvline(0, color='black', lw=0.8, alpha=0.5)
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    
    # ===== BOTTOM ROW: FEATURE TABLE =====
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    
    # Prepare table data (now includes SHAP values since they're not on chart)
    table_data = [[f, customer_values.get(f, "N/A"), f'{aggregated_shap[f]:.4f}'] 
                 for f in sorted_features]
    
    # Create color map for table cells
    cmap = LinearSegmentedColormap.from_list('impact_cmap', ["#FF4560", "white", "#008FFB"])
    max_impact = max(abs(v) for v in sorted_values)
    norm_values = [v/max_impact for v in sorted_values]
    
    # Create table
    table = ax2.table(
        cellText=table_data,
        colLabels=["Feature", "Customer Value", "SHAP Value"],
        loc='center',
        cellLoc='center',
        colWidths=[0.5, 0.3, 0.2],
        cellColours=[['white', 'white', cmap(0.5 + 0.5*n)] for n in norm_values]
    )
    
    # Table styling
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)  # Increase row height
    
    # Header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f7f7f7')
    
    # ===== PREDICTION TITLE =====
    prediction = model.predict(customer_data)[0]
    prediction_proba = model.predict_proba(customer_data)[0][1]
    fig.suptitle(
    f"Customer Churn Analysis "
    f"Prediction: {'Churn' if prediction == 1 else 'No Churn'} "
    f"(Probability: {(prediction_proba if prediction == 1 else (1 - prediction_proba)) * 100:.0f}% chance the customer will "
    f"{'leave' if prediction == 1 else 'stay'}) \n",
    fontsize=16, y=0.98
)
    
    # Final layout adjustment

    plt.subplots_adjust(top=0.94)  # Adjust top spacing
    
    # # ====================== [3. RETURN RESULTS] ======================
    # feature_impacts = []
    # for feature, impact in zip(sorted_features, sorted_values):
    #     value = customer_values.get(feature, "N/A")
    #     if abs(impact) < 0.001:
    #         explanation = f"{feature} ('{value}') has negligible impact."
    #     elif impact > 0:
    #         explanation = f"{feature} ('{value}') increases risk by {abs(impact):.3f}."
    #     else:
    #         explanation = f"{feature} ('{value}') decreases risk by {abs(impact):.3f}."
            
    #     feature_impacts.append({
    #         "feature": feature,
    #         "impact": impact,
    #         "value": value,
    #         "explanation": explanation
    #     })
    
    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": float(prediction_proba),
        "plot": fig,
        # "feature_impacts": feature_impacts,
        "base_value": float(base_value),
        "shap_values": shap_values
    }



