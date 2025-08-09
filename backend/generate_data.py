import pandas as pd
import numpy as np

def generate_data(records = 1000):
    data = {
        'sleep_hours': np.random.normal(6.5, 1.5, records),
        'hrv_score': np.random.randint(20, 80, records),
        'stress_level': np.random.randint(1, 10, records),
        'activity_calories': np.random.randint(300, 1500, records),
    }
    df = pd.DataFrame(data)

    risk_score = (
        (df['sleep_hours'] < 5) * 3 +
        (df['hrv_score'] < 40) * 2 +
        (df['stress_level'] > 7) * 3
    )
    df['is_at_risk'] = (risk_score >= 5).astype(int)
    df['sleep_hours'] = df['sleep_hours'].clip(0, 12)
    df.to_csv('ml/synthetic_health_data.csv', index = False)
    print("Synthetic data generated at backend/synthetic_health_data.csv")

if __name__ == '__main__':
    generate_data()