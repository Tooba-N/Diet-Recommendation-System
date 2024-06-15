#!/usr/bin/env python
# coding: utf-8

# In[162]:


import pandas as pd
df=pd.read_csv("dataset.csv",nrows=60000)
#df = df.sample(n=50000, random_state=42)
df.head()


# In[163]:


df.columns


# In[164]:


df.shape


# In[165]:


df.describe()


# # DATA VISUALIZATION

# ##  Showing top 3 recipes by review count

# In[166]:


import matplotlib.pyplot as plt

# Plot top 10 recipes by review count
top_recipes = df.nlargest(3, 'ReviewCount')
plt.bar(top_recipes['Name'], top_recipes['ReviewCount'])
plt.xlabel('Recipe Name')
plt.ylabel('Review Count')
plt.title('Top 3 Recipes by Review Count')
plt.show()


# ## A scatter plot of calories vs fat content

# In[167]:


plt.scatter(df['Calories'], df['FatContent'])
plt.xlabel('Calories')
plt.ylabel('Fat Content (g)')
plt.title('Relationship between Calories and Fat Content')
plt.show()


# ## Distribution of recipes by category

# In[168]:


plt.bar(recipe_categories.index, recipe_categories.values)
plt.xlabel('Recipe Category')
plt.ylabel('Count')
plt.title('Distribution of Recipes by Category')
plt.xticks(rotation=45, fontsize=8) 
plt.show()


# # Pre Processing

# In[169]:


#Removing Columns

df=df.drop(['AuthorId', 'AuthorName', 'CookTime', 'PrepTime',
       'TotalTime', 'DatePublished','Images','RecipeIngredientQuantities', 'RecipeIngredientParts',
       'AggregatedRating', 'ReviewCount','RecipeServings', 'RecipeYield','Keywords'], axis=1)
df.head()


# ## CHECKING FOR NULL VALUES

# In[170]:


df.isnull().sum()


# ## REMOVING NULL VALUES

# In[171]:


df=df.dropna()


# In[172]:


df.isnull().sum()


# ## CALCULATING AND CATEGORIZING BMI

# In[173]:


# Calculate BMI
def BMI_cal(weight, height):
    return weight / (height ** 2)

# Categorize BMI
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal weight'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'


# ## CALCULATING BMR

# In[174]:


# Calculate BMR
def calculate_bmr(gender, weight, height, age):
    height_cm = height * 100
    if gender == 'Male':
        return 10 * weight + 6.25 * height_cm - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height_cm - 5 * age - 161


# ## CALCULATING DAILY CALORIES INTAKE BASED ON ACTIVITY LEVEL

# In[175]:


# Calculate daily calories based on activity level
def calculate_calories(activity_level, bmr):
    if activity_level == "Sedentary(Not Active)":
        return bmr * 1.2
    elif activity_level == "Moderately Active":
        return bmr * 1.55
    elif activity_level == "Very Active":
        return bmr * 1.725


# ## INPUT VALUES AND CALLING FUCNCTIONS

# In[176]:


# INPUT VALUES
age = 22
weight =54
height = 1.65   #in meter
gender = 'Female'
activity_level="Moderately Active"
plan="Maintain weight"
calorie_weights=1

bmi = BMI_cal(weight, height)
bmi_category = categorize_bmi(bmi)
bmr=calculate_bmr(gender, weight, height, age)
calories=calculate_calories(activity_level, bmr)
daily_calories = int(calories * calorie_weights)
print(f"Your BMI is: {bmi}\nYour BMR is: {bmr}\nRecommended Calories: {daily_calories} per day")


# ## MODEL IMPLEMENTATION

# In[181]:


from sklearn.neighbors import NearestNeighbors
import numpy as np


# Training model
def train_model(df):
    X = df[['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
            'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent',
            'ProteinContent']]
    
    X_columns = X.columns  # Store column names before converting to NumPy array
    X = X.values  # Convert DataFrame to NumPy array
    
    # Fit NearestNeighbors with the NumPy array X
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X)
    
    return nbrs, X_columns

# Generating recommendations
def generate_recommendations(nbrs, daily_nutrition_goals, num_days):
    daily_nutrition_goals = np.array(daily_nutrition_goals)
    meals_nutrition = []

    for day in range(num_days):
        # Simulate daily nutrition goals
        meal_nutrition = np.random.uniform(0.8, 1.2, size=(3, len(daily_nutrition_goals))) * daily_nutrition_goals
        distances, recommended_recipes = nbrs.kneighbors(meal_nutrition.reshape(3, -1))

        recommended_recipe_ids = [int(recipe_id) for recipe_id in recommended_recipes.flatten()]
        meals_nutrition.append(recommended_recipe_ids)

    return meals_nutrition


def get_recipes(recipe_id):
    recipe_row = df.iloc[recipe_id]  
    name = recipe_row['Name']
    recipe_category = recipe_row['RecipeCategory']
    description = recipe_row['Description']
    instructions = recipe_row['RecipeInstructions']
    return name, recipe_category, description, instructions


num_days = 3  
daily_nutrition_goals = [2000, 30, 10, 300, 5, 55, 25, 50, 15]  

# Generate Recipes
nbrs, X_columns = train_model(df)
recommended_recipes_per_day = generate_recommendations(nbrs, daily_nutrition_goals, num_days)




# ## PRINTING RCOMMENDED MEALS

# In[182]:


# Print recommended meal plan
print("### RECOMMENDED MEAL PLAN")

meal_types = ['Breakfast', 'Lunch', 'Dinner']

for day, recipe_ids in enumerate(recommended_recipes_per_day):
    print(f"#### DAY {day + 1}")
    for meal, recipe_id in zip(meal_types, recipe_ids):
        print(f"##### {meal}")
        name, category, description, instructions = get_recipes(recipe_id)
        print(f"**Recipe Name:** {name}")
        print(f"**Category:** {category}")
        print(f"**Description:** {description}")
        print(f"**Instructions:** {instructions}")
        print("---")


# In[ ]:


## EVALUATING THE PERFO


# In[187]:


def evaluate_model(recommended_recipes_per_day, daily_nutrition_goals, df):
    cosine_sim = 0
    recipe_diversity = {}

    for day in recommended_recipes_per_day:
        day_recipes = [get_recipes(recipe_id) for recipe_id in day]
        day_nutrition = np.array([df.iloc[recipe_id][X_columns] for recipe_id in day])

        # Calculate Cosine Similarity
        cosine_sim += np.mean([np.dot(day_nutrition[i], daily_nutrition_goals) / (np.linalg.norm(day_nutrition[i]) * np.linalg.norm(daily_nutrition_goals)) for i in range(len(day))])

        # Calculate Recipe Diversity
        for recipe in day_recipes:
            category = recipe[1]
            if category not in recipe_diversity:
                recipe_diversity[category] = 1
            else:
                recipe_diversity[category] += 1

    cosine_sim /= num_days

    print("Cosine Similarity:", cosine_sim)
    print("Recipe Diversity:", recipe_diversity)

evaluate_model(recommended_recipes_per_day, daily_nutrition_goals, df)


# In[ ]:




