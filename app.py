import pandas as pd
from geopy.distance import great_circle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app)

# ------------------- Load Data Globally -------------------
try:
    members_df = pd.read_csv("dataset/members_large_deterministic.csv")
    providers_df = pd.read_csv("dataset/providers_enhanced.csv")
    print("Successfully loaded members and providers data.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Make sure the CSV files are in the correct directory.")
    members_df = pd.DataFrame()
    providers_df = pd.DataFrame()


# ------------------- ML Pipeline Functions (Integrated) -------------------

def find_providers_in_radius(member_lat, member_lon, providers_df, max_distance_km):
    providers_in_radius = []
    for _, provider in providers_df.iterrows():
        dist_km = great_circle((member_lat, member_lon),
                               (provider['latitude'], provider['longitude'])).km
        if dist_km <= max_distance_km:
            provider_data = provider.to_dict()
            provider_data['distance_km'] = dist_km
            provider_data['distance_miles'] = dist_km * 0.621371
            provider_data['drive_time_minutes'] = (dist_km / 50) * 60  # 50 km/h speed assumption
            providers_in_radius.append(provider_data)
    return pd.DataFrame(providers_in_radius)

def calculate_quality_score(row):
    score, total_weight = 0, 0
    exp_score = min(row.get("experience_years", 0) / 40, 1) * 10
    score += exp_score * 0.20
    total_weight += 0.20
    rating_score = (row.get("patient_rating", 0) / 5) * 10
    score += rating_score * 0.20
    total_weight += 0.20
    cms_score = (row.get("CMS_quality_score", 0) / 5) * 10
    score += cms_score * 0.25
    total_weight += 0.25
    risk_score = (1 - row.get("risk_rate", 0)) * 10
    score += risk_score * 0.15
    total_weight += 0.15
    cert_score = (5 if row.get("certified", False) else 0) + (5 if row.get("background_check_passed", False) else 0)
    score += cert_score * 0.10
    total_weight += 0.10
    tele_score = 10 if row.get("telehealth_available", False) else 0
    score += tele_score * 0.10
    total_weight += 0.10
    final_score = max(1, min(5, score / total_weight))
    return final_score

def quality_model(selected_providers, member):
    quality_df = selected_providers.copy()
    quality_df["telehealth_preference"] = member['telehealth_preference']
    quality_df["quality_score"] = quality_df.apply(calculate_quality_score, axis=1).round(1)
    quality_df["benchmark_percent"] = (2 * quality_df["quality_score"] * 10).round(0).astype(int)
    return quality_df

coverage_map = {"PPO": 0.85, "HMO": 0.75, "EPO": 0.65}
visits_map = {"Low": 2, "Medium": 5, "High": 10}

def calculate_payment(provider, member):
    adj_cost = provider.get("service_cost", 0) * (1 + 0.01 * provider.get("experience_years", 0))
    wait_penalty = 1 + 0.01 * max(0, provider.get("wait_time_days", 0) - member.get("expected_wait_time_days", 5))
    coverage_share = coverage_map.get(member.get("coverage_plan", "PPO"), 0.6)
    visits = visits_map.get(member.get("risk_level", "Medium"), 5)
    base_payment = adj_cost * coverage_share
    member_share = 0.2 * member.get("invested_amount", 0)
    raw_payment = (base_payment - member_share) * visits * wait_penalty
    min_payment = 0.2 * adj_cost * visits
    payment = max(raw_payment, min_payment)
    return payment, member_share

def cost_model(quality_df, member):
    payments = quality_df.apply(lambda p: calculate_payment(p, member), axis=1)
    quality_df["insurance_payment"] = [p[0] for p in payments]
    quality_df["member_share"] = [p[1] for p in payments]
    if 'name' not in quality_df.columns and 'provider_name' in quality_df.columns:
        quality_df.rename(columns={'provider_name': 'name'}, inplace=True)
    return quality_df

def rank_with_specialty_priority(final_df, member, top_n=10):
    primary = member['primary_specialty_needed']
    secondary = member['secondary_specialty_needed']

    def specialty_priority(specialty):
        if specialty == primary:
            return 1
        elif specialty == secondary:
            return 2
        else:
            return 3

    final_df["specialty_priority"] = final_df["specialty"].apply(specialty_priority)
    ranked = final_df.sort_values(
        by=["specialty_priority", "quality_score", "distance_miles", "insurance_payment"],
        ascending=[True, False, True, True]
    )
    return ranked.head(top_n)


# ------------------- Flask Application Routes -------------------

@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/dashboard')
def dashboard():
    return render_template('InputPage.html')


@app.route('/map')
def map_view():
    return render_template('Map.html')


@app.route('/api/get-all-providers')
def get_all_providers():
    if providers_df.empty:
        return jsonify({"error": "Provider data not available"}), 500
    df = providers_df.copy()
    if 'name' not in df.columns and 'provider_name' in df.columns:
        df.rename(columns={'provider_name': 'name'}, inplace=True)
    return jsonify(df.to_dict(orient='records'))


@app.route('/api/find-providers', methods=['POST'])
def find_providers_api():
    data = request.get_json()
    member_id = data.get('member_id')
    if not member_id:
        return jsonify({'message': 'Member ID is required.'}), 400
    if members_df.empty or providers_df.empty:
        return jsonify({'message': 'Server data not available.'}), 500

    member_data = members_df[members_df["member_id"] == member_id]
    if member_data.empty:
        return jsonify({'message': f'Member ID {member_id} not found.'}), 404

    member = member_data.iloc[0].to_dict()

    # --- Execute the full ML Pipeline ---
    geo_df = find_providers_in_radius(member["latitude"], member["longitude"], providers_df, member["max_travel_distance_km"])
    if geo_df.empty:
        return jsonify({
            "providers": [],
            "member_location": {"lat": member['latitude'], "lon": member['longitude']}
        })

    quality_df = quality_model(geo_df, member)
    final_df = cost_model(quality_df, member)
    recommended = rank_with_specialty_priority(final_df, member, top_n=10)

    response_data = {
        "providers": recommended.to_dict(orient='records'),
        "member_location": {"lat": member['latitude'], "lon": member['longitude']}
    }
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='127.0.0.1')

