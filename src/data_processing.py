# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
# Load the dataset
df = pd.read_csv('data/complaints.csv')
df.info()
print("\n=== Product Distribution ===")
product_dist = df['Product'].value_counts(normalize=True)
print(product_dist)
df['Date received'] = pd.to_datetime(df['Date received'])

monthly_complaints = df.groupby(df['Date received'].dt.month)['Complaint ID'].count()

# Plot monthly complaints
plt.figure(figsize=(10, 6))
monthly_complaints.plot(kind='bar', color='skyblue')
plt.title('Monthly Distribution of Consumer Complaints')
plt.xlabel('Month')
plt.ylabel('Number of Complaints')
plt.xticks(rotation=0)
plt.show()
product_issue_counts = df.groupby(['Product', 'Issue']).size().reset_index(name='count')

most_complained_product = product_issue_counts.groupby('Product')['count'].sum().idxmax()
most_complained_product_issues = product_issue_counts[product_issue_counts['Product'] == most_complained_product]

plt.figure(figsize=(10, 6))
most_complained_product_issues.sort_values(by='count', ascending=False, inplace=True)
top_issues = most_complained_product_issues.head(10)  # Display top 10 issues
plt.barh(top_issues['Issue'], top_issues['count'], color='skyblue')
plt.title(f'Most Common Issues for {most_complained_product}')
plt.xlabel('Number of Complaints')
plt.ylabel('Issue')
plt.gca().invert_yaxis()
plt.show()
company_response_counts = df['Company response to consumer'].value_counts()

plt.figure(figsize=(10, 6))
company_response_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Company Responses to Consumer Complaints')
plt.xlabel('Company Response')
plt.ylabel('Number of Complaints')
plt.xticks(rotation=45, ha='right')
plt.show()
untimely_complaints = df[df['Timely response?'] == 'No']
plt.figure(figsize=(10, 6))
untimely_complaints['Product'].value_counts().plot(kind='bar', color='pink')
plt.title('Distribution of Product Types for Untimely Responses')
plt.xlabel('Product Type')
plt.ylabel('Number of Complaints')
plt.xticks(rotation=45, ha='right')
plt.show()
plt.figure(figsize=(10, 6))
untimely_complaints['Issue'].value_counts().head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Issues for Untimely Responses')
plt.xlabel('Issue')
plt.ylabel('Number of Complaints')
plt.xticks(rotation=45, ha='right')
plt.show()
submitted_via_counts = df['Submitted via'].value_counts()
plt.figure(figsize=(8, 6))
submitted_via_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Complaints by Submission Method')
plt.xlabel('Submission Method')
plt.ylabel('Number of Complaints')
plt.xticks(rotation=45, ha='right')
plt.show()
df['Year'] = pd.to_datetime(df['Date received']).dt.year
yearly_complaints = df.groupby('Year').size()
plt.figure(figsize=(10, 6))
plt.plot(yearly_complaints.index, yearly_complaints.values, marker='o', color='skyblue', linestyle='-')
plt.title('Trend of Complaints Received Year-wise')
plt.xlabel('Year')
plt.ylabel('Number of Complaints')
plt.xticks(yearly_complaints.index)
plt.grid(True)
plt.show()
response_counts = df.groupby(['Company response to consumer', 'Timely response?']).size().unstack(fill_value=0)
plt.figure(figsize=(10, 6))
sns.barplot(data=response_counts, palette='pastel')
plt.title('Company Response vs. Timely Response')
plt.xlabel('Company Response')
plt.ylabel('Number of Complaints')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Timely Response', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
product_counts = df['Product'].value_counts()
print(product_counts)
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=product_counts.index, y=product_counts.values)
plt.title('Distribution of Complaints Across Products')
plt.xlabel('Product')
plt.ylabel('Number of Complaints')
plt.xticks(rotation=45)
plt.show()
narrative_counts = df['Consumer complaint narrative'].isnull().value_counts()
print(narrative_counts)

# Visualization
plt.figure(figsize=(6, 4))
sns.barplot(x=narrative_counts.index, y=narrative_counts.values)
plt.title('Complaints with and without Narratives')
plt.xlabel('Has Narrative')
plt.ylabel('Number of Complaints')
plt.xticks(ticks=[0, 1], labels=['Yes', 'No'])
plt.show()
# 2. Calculate and visualize the length of Consumer complaint narrative
df['narrative_length'] = df['Consumer complaint narrative'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10, 6))
sns.histplot(df['narrative_length'], bins=30, kde=True)
plt.title('Distribution of Consumer Complaint Narrative Length')
plt.xlabel('Narrative Length (Word Count)')
plt.ylabel('Frequency')
plt.show()
# 3. Identify number of complaints with and without narratives
narrative_counts = df['Consumer complaint narrative'].notnull().value_counts()
narrative_counts.index = ['Without Narrative', 'With Narrative']
plt.figure(figsize=(6, 4))
sns.barplot(x=narrative_counts.index, y=narrative_counts.values)
plt.title('Complaints With and Without Narratives')
plt.ylabel('Number of Complaints')
plt.show()
# Filter for specified products and remove empty narratives
products_of_interest = ['Credit card', 'Personal loan', 'Buy Now, Pay Later', 'Savings account', 'Money transfers']
filtered_df = df[df['Product'].isin(products_of_interest) & df['Consumer complaint narrative'].notnull()]

# Clean the narrative text
def clean_text(text):
    text = text.lower()  # Lowercase
    text = ''.join(char for char in text if char.isalnum() or char.isspace())  # Remove special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

filtered_df['cleaned_narrative'] = filtered_df['Consumer complaint narrative'].apply(clean_text)
filtered_df.to_csv('../data/filtered_complaints.csv', index=False)