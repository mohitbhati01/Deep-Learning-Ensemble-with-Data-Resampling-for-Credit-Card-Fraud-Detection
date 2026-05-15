import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

def generate_dataset(num_rows=100000, fraud_rate=0.02):
    fake = Faker()
    # Setting seeds for reproducibility
    Faker.seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 1. Generate underlying user profiles 
    # On avg, 20 transactions per user
    num_users = int(num_rows / 20) 
    users = {
        str(uuid.uuid4()): {
            'avg_spend_per_day': max(5, np.random.lognormal(mean=3.5, sigma=1.0)), # Highly right-skewed normal spending
            'card_age': random.randint(10, 3650), # Age in days
            'base_lat': float(fake.latitude()),
            'base_lon': float(fake.longitude())
        }
        for _ in range(num_users)
    }
    
    user_ids = list(users.keys())
    
    # Lists for categorical generation
    categories = ['groceries', 'entertainment', 'utilities', 'electronics', 'travel', 'dining', 'health', 'retail']
    devices = ['mobile', 'web', 'pos']
    transaction_types = ['online', 'offline']
    
    data = []
    
    for i in range(num_rows):
        # Enforce 2% class imbalance
        is_fraud = 1 if random.random() < fraud_rate else 0
        user_id = random.choice(user_ids)
        user = users[user_id]
        
        # Base realistic features
        t_time = fake.date_time_between(start_date="-1y", end_date="now")
        is_foreign = 0
        is_high_risk = 0
        
        if is_fraud:
            # Inject realistic fraud patterns
            fraud_type = random.choice(['sudden_high_amount', 'foreign_high_risk', 'rapid_transactions', 'unusual_night_time'])
            
            if fraud_type == 'sudden_high_amount':
                t_amount = user['avg_spend_per_day'] * random.uniform(10, 50) # Huge multiplier
                m_category = random.choice(categories)
                device = random.choice(devices)
                t_type = random.choice(transaction_types)
                lat, lon = user['base_lat'], user['base_lon'] # Stolen card used locally or online
                prev_gap = random.uniform(10, 100) # minutes
                num_last_24h = random.randint(1, 4)
                
            elif fraud_type == 'foreign_high_risk':
                t_amount = user['avg_spend_per_day'] * random.uniform(1, 5)
                m_category = 'travel'
                device = 'web'
                t_type = 'online'
                is_foreign = 1
                is_high_risk = 1
                lat, lon = float(fake.latitude()), float(fake.longitude()) # Completely different geological location
                prev_gap = random.uniform(1, 60)
                num_last_24h = random.randint(1, 3)
                
            elif fraud_type == 'rapid_transactions':
                t_amount = user['avg_spend_per_day'] * random.uniform(0.5, 3)
                m_category = 'electronics'  # Easy to resell
                device = 'mobile'
                t_type = 'online'
                lat, lon = float(fake.latitude()), float(fake.longitude())
                prev_gap = random.uniform(0.1, 5) # Exceptionally short gap (seconds/mins)
                num_last_24h = random.randint(8, 25) # Huge surge in transactions today
                
            elif fraud_type == 'unusual_night_time': # Assuming timezone context
                t_time = t_time.replace(hour=random.randint(0, 4))
                t_amount = user['avg_spend_per_day'] * random.uniform(2, 8)
                m_category = 'entertainment'
                device = 'web'
                t_type = 'online'
                lat, lon = user['base_lat'], user['base_lon']
                prev_gap = random.uniform(30, 300)
                num_last_24h = random.randint(1, 4)
                
        else:
            # 2. Normal behaviour patterns
            # Introduce natural variance / noise
            amount_multiplier = np.random.lognormal(mean=-0.5, sigma=0.5) 
            t_amount = user['avg_spend_per_day'] * amount_multiplier
            t_amount = min(max(t_amount, 1.0), user['avg_spend_per_day'] * 15) # Cap it so it isn't ridiculous
            
            # Weighted categories mimicking real human behaviour
            m_category = random.choices(categories, weights=[0.3, 0.1, 0.1, 0.05, 0.05, 0.2, 0.05, 0.15])[0]
            
            if random.random() < 0.95:
                # Same general location
                lat = user['base_lat'] + random.uniform(-0.05, 0.05)
                lon = user['base_lon'] + random.uniform(-0.05, 0.05)
            else:
                # Occasionally travels naturally
                lat, lon = float(fake.latitude()), float(fake.longitude())
                is_foreign = 1 if random.random() < 0.2 else 0 
                
            device = random.choice(devices)
            
            # Data feature correlation: Offline usually POS, Online usually web/mobile
            if device == 'pos':
                t_type = random.choices(['offline', 'online'], weights=[0.9, 0.1])[0]
            else:
                t_type = random.choices(['online', 'offline'], weights=[0.8, 0.2])[0]
                
            # Most normal transactions happen during day/evening (7 AM to 11 PM)
            if random.random() < 0.9:
                t_time = t_time.replace(hour=random.randint(7, 23))
                
            # Gaps follow lognormal distribution 
            prev_gap_hours = np.random.lognormal(mean=2, sigma=1) 
            prev_gap = min(prev_gap_hours * 60, 10000) 
            
            num_last_24h = np.random.poisson(lam=1.5) # mostly 1-2, sometimes 0 or 3-5

        # Compile the generated row
        row = {
            'transaction_id': str(uuid.uuid4()),
            'user_id': user_id,
            'transaction_time': t_time,
            'transaction_amount': round(t_amount, 2),
            'merchant_category': m_category,
            'merchant_id': fake.bban(), # Just generic alphanumeric string spoofing
            'location': f"{round(lat, 4)}, {round(lon, 4)}",
            'device_type': device,
            'transaction_type': t_type,
            'previous_transaction_gap': round(prev_gap, 2),
            'avg_spend_per_day': round(user['avg_spend_per_day'], 2),
            'is_foreign_transaction': is_foreign,
            'is_high_risk_country': is_high_risk,
            'number_of_transactions_last_24hrs': num_last_24h,
            'card_age': user['card_age'],
            'fraud_label': is_fraud
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Sort logically so individual user timelines are chronological
    df = df.sort_values(by=['user_id', 'transaction_time']).reset_index(drop=True)
    
    # 3. Save Output
    csv_file = 'synthetic_credit_card_data.csv'
    df.to_csv(csv_file, index=False)
    print(f"Dataset successfully saved to {csv_file}")
    
    # 4. Data Visualization
    try:
        plt.figure(figsize=(10, 6))
        
        # Create the countplot
        ax = sns.countplot(data=df, x='fraud_label', hue='fraud_label', palette=['#2ecc71', '#e74c3c'], legend=False)
        plt.title('Fraud vs Normal Transactions (Target Imbalance: ~98/2)', fontsize=14, pad=15)
        plt.xlabel('Fraud Label (0 = Normal, 1 = Fraud)', fontsize=12)
        plt.ylabel('Number of Transactions', fontsize=12)
        
        # Add count labels and percentages
        total = len(df)
        for p in ax.patches:
            height = p.get_height()
            percentage = f'{100 * height / total:.1f}%'
            ax.annotate(f'{int(height)}\n({percentage})', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5),
                        textcoords='offset points')
                        
        # Adjust y limit to make room for labels
        plt.ylim(0, df['fraud_label'].value_counts().max() * 1.15)
        sns.despine()
        
        img_file = 'fraud_distribution.png'
        plt.savefig(img_file, bbox_inches='tight', dpi=300)
        print(f"Distribution plot saved to {img_file}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
        
    return df

if __name__ == '__main__':
    print("Generating 100,000 realistic transactions... Please wait.")
    df = generate_dataset(num_rows=100000, fraud_rate=0.02)
    print("\n================ SAMPLE DATA (First 5 Rows) ================")
    print(df.head().to_string())
    print("============================================================\n")
