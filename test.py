import json
import pandas as pd
import numpy as np
from datetime import timedelta
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

import openai
from openai.embeddings_utils import get_embedding

openai.api_key = "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"

from azure.kusto.data import KustoClient, KustoConnectionStringBuilder, ClientRequestProperties
from azure.kusto.data.exceptions import KustoServiceError
from azure.kusto.data.helpers import dataframe_from_result_table

cluster = "https://icmcluster.kusto.windows.net"
kcsb = KustoConnectionStringBuilder.with_aad_device_authentication(cluster)
client = KustoClient(kcsb)

db = "IcmDataWarehouse"
query = ("( IncidentHistory | where ChangeDescription startswith \"Upgraded severity\" and Severity "
         "<= 2 | distinct "
         "IncidentId, OwningTenantName, Severity, IsCustomerImpacting, Title | take 10000 ) | join kind=innerunique "
         "Incidents on IncidentId | distinct IncidentId, OwningTenantName, Severity, IsCustomerImpacting, Title1, "
         "IncidentType, SourceName, Escalation = 1")
response = client.execute(db, query)

with open("escalations.json", "w+") as f:
    f.write(str(response.primary_results[0]))

dataframe = dataframe_from_result_table(response.primary_results[0])

print(dataframe)

client.close()

with open('escalations.json', 'r') as f:
    escalations = json.load(f)

escalations_df = pd.DataFrame(escalations)

with open('non-escalations.json', 'r') as f:
    non_escalations = json.load(f)

non_escalations_df = pd.DataFrame(non_escalations)

train_df = pd.concat([escalations_df, non_escalations_df])
train_df = train_df.sample(frac=1)
train_df = train_df.drop_duplicates()
# train_df.to_json('train-data.json', "records")

train_df['combined'] = str(train_df.OwningTenantName).strip() + " " + str(
    train_df.IsCustomerImpacting).strip() + " " + str(train_df.Title1).strip() + " " + str(
    train_df.SourceName).strip()

train_df['n_tokens'] = train_df.combined.apply(lambda x: len(tokenizer.encode(x, truncation=True)))
train_df = train_df[train_df.n_tokens < 8000].tail(1_000)
print(len(train_df))

train_df['ada_similarity'] = train_df.combined.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
train_df.to_csv('escalation_data_with_embeddings_1k.csv')

with open('escalation_data_with_embeddings_1k.csv', 'r') as f:
    train_df = pd.read_csv(f)

train_df["ada_similarity"] = train_df.ada_similarity.apply(eval).apply(np.array)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(list(train_df.ada_similarity.values), train_df.Escalation,
                                                    test_size=0.2,
                                                    random_state=42)

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(X_train, y_train)
preds = rfr.predict(X_test)

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)

print(f"Ada similarity embedding performance on 1k Amazon reviews: mse={mse:.2f}, mae={mae:.2f}")
