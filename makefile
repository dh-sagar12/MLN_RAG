start:
	./run.sh

.ONESHELL:
docker-build-push:
	@read -p "Enter Build Version: " version; \
	docker build -t sagardhakal/mln:$$version \
		-f Dockerfile .; \
	docker push sagardhakal/mln:$$version

up:
	docker compose -f docker-compose.yml up

stop:
	docker compose -f docker-compose.yml down