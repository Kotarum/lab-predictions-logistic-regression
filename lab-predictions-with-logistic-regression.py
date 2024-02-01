
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Getting the data from the sakila database
engine = create_engine('mysql+pymysql://root:password@localhost:3306/sakila')

query = """
SELECT film.film_id, film.title, film.length, film.language_id, film.release_year, film.rental_duration, 
       film.rental_rate, film.rating, rental.rental_date, rental.return_date, rental.customer_id
FROM film
JOIN inventory ON film.film_id = inventory.film_id
JOIN rental ON inventory.inventory_id = rental.inventory_id
"""

df = pd.read_sql(query, engine)

engine.dispose()

# ---------------------------------------------------------------

# Check how is the data
df.head() 
df.info() 

# ---------------------------------------------------------------

# Transform the data
df_encoded = pd.get_dummies(df, columns=['language_id', 'rating'], drop_first=True)


# Scale the data to feed it to the algorithm
scaler = StandardScaler()
numerical_columns = ['length', 'rental_duration', 'rental_rate']
df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])


# Get the data from point 4.
query_last_month = """
SELECT film.film_id, film.title,
       CASE WHEN EXISTS (
           SELECT 1 FROM rental
           WHERE rental.inventory_id = inventory.inventory_id
           AND rental.return_date >= CURDATE() - INTERVAL 1 MONTH
           AND rental.return_date < CURDATE()
       ) THEN 1 ELSE 0 END AS rented_last_month
FROM film
LEFT JOIN inventory ON film.film_id = inventory.film_id
"""

df_last_month = pd.read_sql(query_last_month, engine)

# ---------------------------------------------------------------
# Let's train!

X = df_encoded.drop(columns=['rented_last_month'])
y = df_last_month['rented_last_month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
