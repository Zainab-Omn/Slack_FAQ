FROM python:3.12-slim


WORKDIR /app


# System deps (add others if your libs need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential \
&& rm -rf /var/lib/apt/lists/*


# Install pipenv and project deps (system site-packages)
RUN pip install --no-cache-dir pipenv
COPY Pipfile Pipfile.lock* ./
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy --system


# Copy code
COPY . .


# Streamlit configs
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
EXPOSE 8501


# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]