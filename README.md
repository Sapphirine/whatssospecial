# What's so Special

## What's so special? (Team 44)

What's so special has 2 different attributes:
  1. What's so special in a business?
  2. What's sp special at a particular location?
  
# Dataset and Analysis:
We are using Yelp dataset which has about 6.5 million reviews for around 190k businesses. The data is available in the json format.
Based on initial analysis to find what's so special in a business, we obtained the following results:

What's so special in the business named " Emerald Chinese Restaurant"?:

![Emerald Chinese](dimsum.jpg)
This clearly shows that dim sum is the speciality of this restaurant

What's so special in the business named " Fremont Arcade"?:

![Fremont arcade](pinball machine.png)
This clearly shows that pinball machine is the speciality of this place

Similarly, What's so special at the location "New Market"?:

![Emerald Chinese](newmarket.jpg)
This shows that Restaurants are the speciality of this location

Our software architecture uses HTML, CSS, JavaScript for frontend user interface. We also used Django for our web application development. Behind the scenes, the large public Yelp dataset is stored in Google Cloud Storage and the analytics on the data has been done using PySpark and Python. Our analysis results are stored in Google BigQuery and are queried upon request from SQL queries in our Django application. In each of these softwares, we have used software libraries to ease our development process. For the natural language processing techniques used for Yelp reviews analysis, we used Python NLTK libraries.
  

