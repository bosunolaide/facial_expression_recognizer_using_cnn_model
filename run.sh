#!/bin/bash
set -e
python -m app.api &
exec streamlit run streamlit_app/app_streamlit.py --server.port 8501 --server.address 0.0.0.0