import requests
import os
import csv
import time
import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_URL = "https://api.github.com"
TOKEN = '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'
headers = {'Authorization': f'token {TOKEN}'}

def fetch_users():
    users = []
    page = 1

    while True:
        response = requests.get(f"{BASE_URL}/search/users?q=location:Barcelona+followers:>100&per_page=100&page={page}", headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch users: {response.json()}")
            break
        
        data = response.json()
        users.extend(data['items'])

        if 'next' not in response.links:
            break

        page += 1
        time.sleep(1)

    return users

def fetch_user_details(user_login):
    response = requests.get(f"{BASE_URL}/user/{user_login}", headers=headers)
    return response.json() if response.status_code == 200 else None

def fetch_user_repos(user_login):
    repos = []
    page = 1

    while True:
        response = requests.get(f"{BASE_URL}/users/{user_login}/repos?per_page=100&page={page}", headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch repositories for {user_login}: {response.json()}")
            break
        
        data = response.json()
        repos.extend(data)

        if len(data) < 100:
            break

        page += 1
        time.sleep(1)

    return repos

def clean_company(company):
    if company:
        return company.strip().lstrip('@').upper()
    return ""

def main():
    if not os.path.exists('users.csv'):
        users = fetch_users()
        print(len(users))
        
        user_data = []
        count = 1
        for user in users:
            print("user info", count)
            details = fetch_user_details(user['id'])
            print("user info", count, "done")
            count += 1
            if details:
                user_data.append({
                    "login": details['login'],
                    "name": details.get('name', ''),
                    "company": clean_company(details.get('company', '')),
                    "location": details.get('location', ''),
                    "email": details.get('email', ''),
                    "hireable": details.get('hireable', False),
                    "bio": details.get('bio', ''),
                    "public_repos": details.get('public_repos', 0),
                    "followers": details.get('followers', 0),
                    "following": details.get('following', 0),
                    "created_at": details.get('created_at', '')
                })
            time.sleep(1)
        print(len(user_data))
        

        # Write to users.csv
        with open('users.csv', 'w', newline='') as csvfile:
            fieldnames = user_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(user_data)

    if not os.path.exists('repositories.csv'):
        repo_data = []
        count = 1
        for user in user_data:
            print("repo info", count)
            repos = fetch_user_repos(user['login'])
            print("repo info", count, "done")
            count += 1
            for repo in repos:
                repo_data.append({
                    "login": user['login'],
                    "full_name": repo['full_name'],
                    "created_at": repo['created_at'],
                    "stargazers_count": repo['stargazers_count'],
                    "watchers_count": repo['watchers_count'],
                    "language": repo['language'],
                    "has_projects": repo.get('has_projects', False),
                    "has_wiki": repo.get('has_wiki', False),
                    "license_name": repo.get('license_name', "")
                })
            time.sleep(1)

        with open('repositories.csv', 'w', newline='') as csvfile:
            fieldnames = repo_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(repo_data)
    
    users_df = pd.read_csv('users.csv')
    repos_df = pd.read_csv('repositories.csv')

    users_df['created_at'] = pd.to_datetime(users_df['created_at'], errors='coerce')

    # 1. Top 5 users in Barcelona with the highest number of followers
    barcelona_users = users_df[users_df['location'].str.contains('Barcelona', na=False)]
    top_users = barcelona_users.nlargest(5, 'followers')['login'].tolist()
    print(1, ', '.join(top_users))

    # 2. 5 earliest registered GitHub users in Barcelona
    earliest_users = barcelona_users.nsmallest(5, 'created_at')['login'].tolist()
    print(2, ', '.join(earliest_users))

    # 3. 3 most popular licenses among these users
    barcelona_repos = repos_df[repos_df['login'].isin(barcelona_users['login'])]
    popular_licenses = barcelona_repos['license_name'].dropna().value_counts().nlargest(3).index.tolist()
    print(3, ', '.join(popular_licenses))

    # 4. Majority company among these developers
    majority_company = barcelona_users['company'].mode()[0]
    print(4, majority_company)

    # 5. Most popular programming language among these users
    popular_language = repos_df[repos_df['login'].isin(barcelona_users['login'])]['language'].mode()[0]
    print(5, popular_language)

    # 6. Second most popular programming language among users who joined after 2020
    recent_users = users_df[users_df['created_at'] > '2020-01-01']
    recent_popular_language = repos_df[repos_df['login'].isin(recent_users['login'])]['language'].value_counts().nlargest(2).index.tolist()[1]
    print(6, recent_popular_language)

    # 7. Language with the highest average number of stars per repository
    avg_stars_language = repos_df.groupby('language')['stargazers_count'].mean().idxmax()
    print(7, avg_stars_language)

    # 8. Top 5 users in terms of leader_strength
    barcelona_users['leader_strength'] = barcelona_users['followers'] / (1 + barcelona_users['following'])
    top_leader_strength = barcelona_users.nlargest(5, 'leader_strength')['login'].tolist()
    print(8, ', '.join(top_leader_strength))

    # 9. Correlation between followers and number of public repositories
    correlation = barcelona_users['followers'].corr(barcelona_users['public_repos'])
    print(9, f"{correlation:.3f}")

    # 10. Regression slope of followers on repos
    X = barcelona_users[['public_repos']]
    y = barcelona_users['followers']
    model = LinearRegression().fit(X, y)
    slope_repos = model.coef_[0]
    print(10, f"{slope_repos:.3f}")

    # 11. Correlation between projects and wiki enabled
    correlation_projects_wiki = repos_df['has_projects'].corr(repos_df['has_wiki'])
    print(11, f"{correlation_projects_wiki:.3f}")

    # 12. Average following per user for hireable=true minus average for the rest
    avg_hireable = users_df[users_df['hireable'] == True]['following'].mean()
    avg_non_hireable = users_df[users_df['hireable'] == False]['following'].mean()
    difference_hireable = avg_hireable - avg_non_hireable
    print(12, f"{difference_hireable:.3f}")

    # 13. Correlation of bio length with followers
    users_df['bio_length'] = users_df['bio'].str.split().str.len()
    correlation_bio_followers = users_df.dropna(subset=['bio'])['bio_length'].corr(users_df['followers'])
    print(13, f"{correlation_bio_followers:.3f}")

    # 14. Top 5 users who created the most repositories on weekends (UTC)
    repos_df['created_at'] = pd.to_datetime(repos_df['created_at'], errors='coerce')
    repos_df['weekday'] = repos_df['created_at'].dt.dayofweek
    weekend_repos = repos_df[repos_df['weekday'] >= 5]
    top_weekend_users = weekend_repos['login'].value_counts().nlargest(5).index.tolist()
    top_weekend_logins = users_df[users_df['login'].isin(top_weekend_users)]['login'].tolist()
    print(14, ', '.join(top_weekend_logins))

    # 15. Fraction of users with email when hireable=true minus fraction of users with email for the rest
    fraction_hireable = users_df[users_df['hireable'] == True]['email'].notna().mean()
    fraction_non_hireable = users_df[users_df['hireable'] == False]['email'].notna().mean()
    email_difference = fraction_hireable - fraction_non_hireable
    print(15, f"{email_difference:.3f}")

    # 16. Most common surname
    users_df['surname'] = users_df['name'].str.split().str[-1].str.strip()
    common_surname = users_df['surname'].mode()
    common_surname_count = users_df['surname'].value_counts().max()
    common_surnames = ', '.join(common_surname.sort_values().tolist())
    print(16, f"{common_surnames} ({common_surname_count})")


if __name__ == "__main__":
    main()