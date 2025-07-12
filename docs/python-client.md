# Python client for MOTIS

## Install dependencies
```sh
pip install openapi-python-client
```

## Generate Python code from OpenAPI specifications
```sh
openapi-python-client generate --path openapi.yaml --output-path motis_api_client --meta none
```

## Use code (example)
```python
from motis_api_client import Client
from motis_api_client.api.routing import one_to_all

with Client(base_url='http://localhost:8080') as client:
  res = one_to_all.sync(one='52.520806, 13.409420', max_travel_time=30, client=client)

res
```
