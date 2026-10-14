import re
import numpy as np
import pandas as pd

def sample_data():
    """
    Generate synthetic B2B lead dataset:
    - Exactly 6,000 records
    - Exactly 25 columns
    - Target 'Converted': exactly 4,500 'No' and 1,500 'Yes'
    - Realistic inconsistent categories (Retail/RETAIL/retail, Paid Ads/Paid ads/Paid_Ads, India/INDIA/ India )
    - Missing numerical and categorical values
    - Preserved canonical 'lead_id'
    """
    r = np.random.default_rng(42)
    n = 6000
    
    # 1. Target column: exactly 4,500 'No' and 1,500 'Yes'
    target_labels = np.array(['No'] * 4500 + ['Yes'] * 1500)
    # Shuffle indices deterministically
    shuffled_idx = r.permutation(n)
    target = target_labels[shuffled_idx]
    y_binary = (target == 'Yes').astype(int)
    
    # 2. Canonical lead identifier
    lead_ids = [f'LEAD-{i+1:05d}' for i in range(n)]
    
    # 3. Categorical features with controlled inconsistency
    # Inconsistent industries (Retail, RETAIL, retail, etc.)
    base_industries = ['Retail', 'Healthcare', 'Restaurant', 'Education', 'Financial Services', 'Technology', 'Professional Services']
    raw_industries = []
    for i in range(n):
        ind = r.choice(base_industries)
        if ind == 'Retail':
            raw_industries.append(r.choice(['Retail', 'RETAIL', 'retail', ' Retail ']))
        elif ind == 'Technology':
            raw_industries.append(r.choice(['Technology', 'TECHNOLOGY', 'technology', 'Tech']))
        elif ind == 'Financial Services':
            raw_industries.append(r.choice(['Financial Services', 'Financial_Services', 'financial services']))
        else:
            raw_industries.append(ind if r.random() > 0.1 else ind.lower())
            
    # Inconsistent lead sources (Paid Ads, Paid ads, Paid_Ads, etc.)
    base_sources = ['Paid Ads', 'Referral', 'Organic Search', 'Cold Outreach', 'Webinar', 'Directory']
    raw_sources = []
    for i in range(n):
        src = r.choice(base_sources)
        if src == 'Paid Ads':
            raw_sources.append(r.choice(['Paid Ads', 'Paid ads', 'Paid_Ads', 'paid ads', ' Paid Ads ']))
        elif src == 'Organic Search':
            raw_sources.append(r.choice(['Organic Search', 'Organic_Search', 'organic search']))
        else:
            raw_sources.append(src)
            
    # Inconsistent countries (India, INDIA, values with extra spaces)
    base_countries = ['India', 'United States', 'United Kingdom', 'Singapore', 'Canada', 'Australia']
    raw_countries = []
    for i in range(n):
        c = r.choice(base_countries)
        if c == 'India':
            raw_countries.append(r.choice(['India', 'INDIA', ' India ', 'india']))
        elif c == 'United States':
            raw_countries.append(r.choice(['United States', 'USA', ' United States ']))
        else:
            raw_countries.append(c)

    # City & State mapping
    city_state_map = {
        'Mumbai': ('Maharashtra', 'India'),
        'Delhi': ('Delhi', 'India'),
        'Bengaluru': ('Karnataka', 'India'),
        'Pune': ('Maharashtra', 'India'),
        'Hyderabad': ('Telangana', 'India'),
        'Chennai': ('Tamil Nadu', 'India'),
        'New York': ('New York', 'United States'),
        'San Francisco': ('California', 'United States'),
        'London': ('Greater London', 'United Kingdom'),
        'Singapore': ('Central', 'Singapore')
    }
    cities = list(city_state_map.keys())
    chosen_cities = r.choice(cities, n)
    chosen_states = [city_state_map[c][0] for c in chosen_cities]
    
    # 4. Realistic business names
    prefixes = ['Apex', 'Global', 'Prime', 'NextGen', 'BlueSky', 'Summit', 'Urban', 'Metro', 'Vanguard', 'Horizon', 'Lotus', 'Green']
    suffixes = ['Solutions', 'Group', 'Enterprises', 'Ventures', 'Technologies', 'Services', 'Partners', 'Labs', 'Hub']
    business_names = [f'{prefixes[i % len(prefixes)]} {suffixes[(i * 7) % len(suffixes)]} {i+1}' for i in range(n)]

    # 5. Continuous signals with correlation to target outcome
    # Higher rating, review count, visits, email opens, and budget correlate with conversion
    rating_noise = r.normal(0, 0.4, n)
    ratings = np.clip(3.2 + 0.9 * y_binary + rating_noise, 1.0, 5.0)
    rating_strings = [f'{v:.1f} stars' if i % 15 == 0 else f'{v:.1f}' for i, v in enumerate(ratings)]
    
    reviews = np.clip((np.exp(r.normal(3.5 + 0.6 * y_binary, 0.8, n))).astype(int), 1, 5000)
    review_strings = [f'{v:,}' if i % 5 == 0 else str(v) for i, v in enumerate(reviews)]
    
    annual_revenue = np.round(np.exp(r.normal(12.5 + 0.5 * y_binary, 1.0, n)), -2)
    employee_count = np.clip(np.exp(r.normal(3.2 + 0.4 * y_binary, 0.9, n)).astype(int), 2, 2500)
    tech_budget = np.round(annual_revenue * np.clip(r.normal(0.04 + 0.02 * y_binary, 0.015, n), 0.005, 0.20), -1)
    company_age = np.clip(r.integers(1, 30, n), 1, 35)

    website_visits = np.clip(r.poisson(15 + 25 * y_binary, n), 0, 500)
    email_opens = np.clip(r.poisson(3 + 8 * y_binary, n), 0, 80)
    content_downloads = np.clip(r.poisson(1 + 3 * y_binary, n), 0, 30)
    
    # Contact & Engagement
    web_presence = r.random(n) < (0.55 + 0.3 * y_binary)
    websites = [f'https://company{i+1}.com' if web_presence[i] else None for i in range(n)]
    
    phones = [f'+9198{i:08d}' if i % 23 != 0 else 'invalid_phone' for i in range(n)]
    emails = [f'contact@company{i+1}.com' if i % 31 != 0 else 'bad-email-format' for i in range(n)]
    
    titles = r.choice(['CEO', 'CTO', 'VP Sales', 'Marketing Director', 'Managing Director', 'Operations Manager', 'Founder'], n)
    
    prev_contact = np.where(r.random(n) < (0.25 + 0.35 * y_binary), 'Yes', 'No')
    prev_response = np.where((prev_contact == 'Yes') & (r.random(n) < (0.3 + 0.45 * y_binary)), 'Yes', 'No')
    social_presence = np.where(r.random(n) < (0.6 + 0.2 * y_binary), 'Yes', 'No')
    
    descriptions = [
        f'B2B {raw_industries[i].strip()} firm operating in {chosen_cities[i]}, offering services to enterprise clients.'
        for i in range(n)
    ]
    
    # Build complete DataFrame with exactly 25 columns
    df = pd.DataFrame({
        'lead_id': lead_ids,                                   # 1. Identifier
        'company_name': business_names,                        # 2. Business Name
        'industry': raw_industries,                            # 3. Categorical (Retail, RETAIL, retail)
        'country': raw_countries,                              # 4. Categorical (India, INDIA, ' India ')
        'city': chosen_cities,                                 # 5. City
        'state': chosen_states,                                # 6. State
        'lead_source': raw_sources,                            # 7. Lead Source (Paid Ads, Paid_Ads)
        'annual_revenue': annual_revenue,                      # 8. Numeric
        'employee_count': employee_count,                      # 9. Numeric
        'tech_budget': tech_budget,                            # 10. Numeric
        'rating': rating_strings,                              # 11. Rating
        'review_count': review_strings,                        # 12. Review Count
        'website': websites,                                   # 13. Website
        'phone': phones,                                       # 14. Phone
        'email': emails,                                       # 15. Email
        'decision_maker_title': titles,                        # 16. Categorical
        'company_age': company_age,                            # 17. Business Age
        'website_visits': website_visits,                      # 18. Numeric
        'email_opens': email_opens,                            # 19. Numeric
        'content_downloads': content_downloads,                # 20. Numeric
        'previous_contact': prev_contact,                      # 21. Boolean
        'previous_response': prev_response,                    # 22. Boolean
        'social_presence': social_presence,                    # 23. Boolean
        'lead_notes': descriptions,                            # 24. Free text
        'Converted': target                                    # 25. Target (4500 No, 1500 Yes)
    })
    
    # Inject realistic missing values across non-target columns (~4% missingness)
    for col in df.columns:
        if col not in ['lead_id', 'Converted', 'company_name']:
            mask = r.random(n) < 0.04
            df.loc[mask, col] = np.nan
            
    return df