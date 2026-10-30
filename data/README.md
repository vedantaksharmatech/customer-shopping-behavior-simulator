# Dataset setup

Download the Retailrocket E-commerce Dataset manually from [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) after signing in. Extract the archive and copy its `events.csv` file to `data/raw/events.csv`. This project does not download data or need Kaggle credentials.

The loader checks the actual CSV header. It requires `timestamp`, `visitorid`, and `event`; `itemid` and `transactionid` are optional. A missing file gives a setup message. You can pass another CSV path with `--data path\to\events.csv`.

Raw data is ignored by Git and is not changed by the program. Do not commit or share the downloaded dataset unless you have the right to do so.
