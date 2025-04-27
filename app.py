import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
df = pd.read_csv('dataset.csv', nrows=50000)

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

# Calculate BMR
def calculate_bmr(gender, weight, height, age):
    height_cm = height * 100
    if gender == 'Male':
        return 10 * weight + 6.25 * height_cm - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height_cm - 5 * age - 161

# Calculate daily calories based on activity level
def calculate_calories(activity_level, bmr):
    if activity_level == "Sedentary(Not Active)":
        return bmr * 1.2
    elif activity_level == "Moderately Active":
        return bmr * 1.55
    elif activity_level == "Very Active":
        return bmr * 1.725

# Function to remove columns and drop missing values
def preprocess_data(df):
    df = df.drop(['AuthorId', 'AuthorName', 'CookTime', 'PrepTime',
                  'TotalTime', 'DatePublished', 'RecipeIngredientQuantities',
                  'RecipeIngredientParts', 'AggregatedRating', 'ReviewCount',
                  'RecipeServings', 'RecipeYield', 'Keywords'], axis=1)
    return df.dropna()

# Function to train the Nearest Neighbors model
def train_model(df):
    X = df[['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
            'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent',
            'ProteinContent']]

    # Fit Nearest Neighbors model
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X)
    return nbrs, X

# Function to generate recommendations
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

# Function to get recipe details
def get_recipes(recipe_id):
    recipe_row = df.iloc[recipe_id]  # Adjusted to use iloc for index-based retrieval
    name = recipe_row['Name']
    recipe_category = recipe_row['RecipeCategory']
    description = recipe_row['Description']
    instructions = recipe_row['RecipeInstructions']
    return name, recipe_category, description, instructions

# Main function to run the Streamlit app
def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'
    if 'recipes_generated' not in st.session_state:
        st.session_state.recipes_generated = False

    if st.session_state.page == 'Home':
        homepage()
    elif st.session_state.page == 'Main':
        main_page()

# Homepage
def homepage():
    st.title("Welcome to the Diet Recommendation System")
    st.image('background.jpg', use_column_width=True)
    st.write(
        "This app helps you find the best diet plan based on your age, gender, weight, height, and activity level.")
    if st.button('Get Started'):
        st.session_state.page = 'Main'

# Main Page
def main_page():
    st.title("Diet Recommendation System")

    age = st.number_input('Age:', 1, 100)
    weight = st.number_input('Weight (Kg):', min_value=0.0, step=1.0)
    height = st.number_input('Height (m):', min_value=0.0, step=0.01)
    st.divider()
    gender = st.radio('Gender:', ['Male', 'Female'])

    if st.button("Calculate BMI"):
        if weight > 0 and height > 0:
            bmi = BMI_cal(weight, height)
            st.session_state.bmi_category = categorize_bmi(bmi)
            container1 = st.container(border=True)
            container1.write(f"## Your BMI is: {bmi:.2f}")
            container1.write(f"## Your BMI category is: {st.session_state.bmi_category}")

        else:
            st.write("Please enter valid weight and height.")

    if 'bmi_category' in st.session_state and st.session_state.bmi_category:
        st.divider()
        st.write("### ACTIVITY LEVEL:")
        activity_level = st.radio("Select appropriate category",
                                  ["Sedentary(Not Active)", "Moderately Active", "Very Active"],
                                  index=None)
        bmr = calculate_bmr(gender, weight, height, age)
        daily_calories = calculate_calories(activity_level, bmr)

        plans = ["Maintain weight", "Mild weight loss", "Weight loss", "Extreme weight loss"]
        weights = [1, 0.9, 0.8, 0.6]
        losses = ['-0 kg/week', '-0.25 kg/week', '-0.5 kg/week', '-1 kg/week']

        st.divider()
        st.title("Calorie Calculator")
        st.write("Select a plan:")
        plan = st.selectbox("Plan", plans)
        if plan:
            index = plans.index(plan)
            daily_calories = int(daily_calories * weights[index])
            container2 = st.container(border=True)
            container2.write(f"Number of calories per day: {daily_calories}")
            container2.write(f"Expected weight loss: {losses[index]}")

        # Select number of days for meal planning
        st.divider()
        num_days = st.selectbox("Select number of days for meal planning", [1, 2, 3])

        # Generate Recipes Button
        if st.button("Generate Recipes"):
            st.session_state.recipes_generated = True
            df_preprocessed = preprocess_data(df)
            nbrs, X = train_model(df_preprocessed)
            daily_nutrition_goals = [daily_calories, 30, 10, 300, 5, 55, 25, 50, 15]  # Example nutrition goals
            recommended_recipes_per_day = generate_recommendations(nbrs, daily_nutrition_goals, num_days)
            st.session_state.recommended_recipes_per_day = recommended_recipes_per_day

    if st.session_state.recipes_generated and 'recommended_recipes_per_day' in st.session_state:
        st.divider()
        st.write("### RECOMMENDED MEAL PLAN")

        meal_types = ['Breakfast', 'Lunch', 'Dinner']
        recommended_recipes_per_day = st.session_state.recommended_recipes_per_day

        for day, recipe_ids in enumerate(recommended_recipes_per_day):
            st.write(f"#### DAY {day + 1}")
            for meal, recipe_id in zip(meal_types, recipe_ids):
                st.write(f"##### {meal}")
                name, category, description, instructions = get_recipes(recipe_id)
                st.write(f"**Recipe Name:** {name}")
                st.write(f"**Category:** {category}")
                st.write(f"**Description:** {description}")
                st.write(f"**Instructions:** {instructions}")
                st.write("---")

if __name__ == "__main__":
    main()

