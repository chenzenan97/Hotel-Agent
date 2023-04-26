.PHONY: run run-container gcloud-deploy

run:
	@streamlit run app.py --server.port=1997 --server.address=0.0.0.0

run-container:
	@docker build . -t app.py
	@docker run -p 8080:8080 app.py
gcloud-deploy:
	@gcloud config set project aipi540-384816
	@gcloud app deploy app.yaml --stop-previous-version
