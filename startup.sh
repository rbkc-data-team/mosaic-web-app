#!/bin/bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000


# #!/bin/bash
# uvicorn api:app --host 0.0.0.0 --port 8000 -workers 2
# streamlit run app.py --server.port 8000 --server.address 0.0.0.0

# python -m uvicorn api:app --host 0.0.0.0 --port 8080 --workers 4
# python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0
# gunicorn api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 s