.PHONY: generate.api

# get git version tag
GIT_VERSION := $(shell git describe --tags --always)
AUTHOR := davidliyutong

#generate.api:
#	rm -rf src/client && \
#	openapi-generator-cli generate -g python -i http://127.0.0.1:8080/openapi.json --skip-validate-spec -o src/client && \

generate.api:
	rm -rf src/client && rm -rf fast-api-client && \
	openapi-python-client generate --url http://127.0.0.1:8080/openapi.json && \
	mv fast-api-client/fast_api_client src/client/

buildx.docker:
	docker buildx build --platform=linux/amd64,linux/arm64 -t ${AUTHOR}/unifolding-backend:${GIT_VERSION} -t ${AUTHOR}/unifolding-backend:latest -f manifests/docker/backend/Dockerfile .
	docker buildx build --platform=linux/amd64,linux/arm64 -t ${AUTHOR}/unifolding-watcher:${GIT_VERSION} -t ${AUTHOR}/unifolding-watcher:latest -f manifests/docker/watcher/Dockerfile .

build.docker:
	docker build -t ${AUTHOR}/unifolding-backend:${GIT_VERSION} -t ${AUTHOR}/unifolding-backend:latest -f manifests/docker/backend/Dockerfile --network host .
	docker build -t ${AUTHOR}/unifolding-watcher:${GIT_VdERSION} -t ${AUTHOR}/unifolding-watcher:latest -f manifests/docker/watcher/Dockerfile --network host .

pushx.docker:
	docker buildx build --push --platform=linux/amd64,linux/arm64 -t ${AUTHOR}/unifolding-backend:${GIT_VERSION} -t ${AUTHOR}/unifolding-backend:latest -f manifests/docker/backend/Dockerfile .
	docker buildx build --push --platform=linux/amd64,linux/arm64 -t ${AUTHOR}/unifolding-watcher:${GIT_VERSION} -t ${AUTHOR}/unifolding-watcher:latest -f manifests/docker/watcher/Dockerfile .