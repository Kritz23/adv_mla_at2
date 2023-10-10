adv_mla_at2
==============================

This is a github repository for the Advance MLA Assessment 2. 

We were adviced to develop two models namely:
1. Predictive Sales Revenue Model: This model uses machine learning to predict the revenue made from sales for a specific item with its sell price, in a specific store on the given date. It makes it possible to manage inventory, plan sales, and make the most money possible. <br>
2. Forecasting Sales Revenue Model: This model uses time series analysis to forecast how much revenue will be made from sales of all items and stores in the next seven days. It gives us useful information for making quick decisions and allocating resources.

These models were developed and deployed on Heroku app: [here](https://still-river-01922-033c48cad951.herokuapp.com/)

----------------

## Run this code:

### Set up environment using poetry

1. Clone this repository and go to the directory by: `cd adv_mla_at2`
2. Initialise the poerty project file: `poetry init`
3. Activate environment: `poetry shell`
4. Install dependencies using: `poetry install`

### Play around with notebooks
1. [Predictive modelling](notebooks/predictive/dhawale_kritika-24587661-sgd_pipeline.ipynb)
2. [Forecasting](notebooks/forecasting/dhawale_kritika-24587661-arima.ipynb)

### Learn about the src files
1. [filter_items.py](src/data/filter_items.py) - Used to filter out data; keep items having more than 10% of the overall sales.
2. [merge.py](src/data/merge.py) - Used to integrate item sell price, date from calendar and calculate revenue using the units sold.
3. [buil_features.py](src/features/build_features.py) - Script to extract date features such as weekday, month and year.
4. [train_model.py](src/models/predictive/train_model.py) - Helper function used to build the SGD pipeline and fit the prediction model.
5. [predict_model.py](src/models/predictive/predict_model.py) - Helper script that takes new data and extracts features of date and gives prediction on the new data.

### Deploy the model in a docker container
1. `cd api`
2. `docker build -t fastapi:latest .`
3. `docker run -dit --rm --name adv_mla_at2_fastapi -p 8080:80 fastapi:latest`

### Release the model to Heroku using container Registry
1. Login with CLI: `heroku login`
2. Login to container registry: `docker ps`
3. `heroku container:login`
4. Push Docker-based app: `heroku container:push web`
5. Deploy the changes: `heroku container:release web`

### Access the API endpoints:
Try this out to predict revenue for an item with its sell price, at a given store and for a date: [predictive api](https://still-river-01922-033c48cad951.herokuapp.com/sales/stores/items?item_id=HOBBIES_1_060&store_id=CA_1&sell_price=30.98&date=2012-11-12') <br>

[/sales/national/](https://still-river-01922-033c48cad951.herokuapp.com/sales/national) will give you the total revenue for the next seven days across all stores for all items.

### Additional notes:
You can change the expected forecast revenue for any number of days you want. Just change the value of `n_periods` in the `arima_pipe.predict(n_periods=7)`. It is in the forecast function of the `main.py` file. 