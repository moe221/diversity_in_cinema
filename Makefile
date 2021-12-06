# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* diversity_in_cinema/*.py

black:
	@black scripts/* diversity_in_cinema/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr diversity_in_cinema-*.dist-info
	@rm -fr diversity_in_cinema.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# ----------------------------------
#      Create A Bucket in GCP
# ----------------------------------

# project id - replace with your GCP project id
PROJECT_ID=le-wagon-bootcamp-328018

# bucket name - replace with your GCP bucket name
BUCKET_NAME=diversity-in-cinema-735

# choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=europe-west1

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}


# ----------------------------------
#      Upload trainig data to GCP
# ----------------------------------

# path to the file to upload to GCP (the path to the file should be absolute or should match the directory where the make command is ran)
# replace with your local path to the `train_1k.csv` and make sure to put the path between quotes
LOCAL_PATH="./raw_data/fairface"

# bucket directory in which to store the uploaded file (`data` is an arbitrary name that we choose to use)
BUCKET_FOLDER=data/training_data

upload_data:
    # @gsutil cp train_1k.csv gs://wagon-ml-my-bucket-name/data/train_1k.csv
	@gsutil -m cp -R ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}


##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME=diversity_in_cinema
FILENAME_GENDER=trainer_gender
FILENAME_RACE=trainer_race
FILENAME_PIPELINE=data
FILENAME_Face=face_averaging_main


##### Job - - - - - - - - - - - - - - - - - - - - - - - - -

JOB_NAME_1=diversity_in_cinema_training_pipeline_gender_$(shell date +'%Y%m%d_%H%M%S')
JOB_NAME_2=diversity_in_cinema_training_pipeline_race_$(shell date +'%Y%m%d_%H%M%S')
JOB_NAME_3=diversity_in_cinema_data_pipeline_1_$(shell date +'%Y%m%d_%H%M%S')
JOB_NAME_4=diversity_in_cinema_data_pipeline_face_$(shell date +'%Y%m%d_%H%M%S')



run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}


# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER = 'trainings'
BUCKET_PIPELINE_FOLDER = "morphing"
PYTHON_VERSION=3.7
RUNTIME_VERSION=1.15

gcp_submit_training_gender_model:
	gcloud ai-platform jobs submit training ${JOB_NAME_1} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME_GENDER} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--scale-tier=CUSTOM \
		--master-machine-type=n1-highmem-8 \
		--stream-logs

gcp_submit_training_race_model:
	gcloud ai-platform jobs submit training ${JOB_NAME_2} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME_RACE} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--scale-tier=CUSTOM \
		--master-machine-type=n1-highmem-8 \
		--stream-logs

gcp_submit_data_scraping_pipeline:
	gcloud ai-platform jobs submit training ${JOB_NAME_3} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_PIPELINE_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME_PIPELINE} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--scale-tier=CUSTOM \
		--master-machine-type=n1-highmem-8 \
		--stream-logs

gcp_submit_data_face_pipeline:
	gcloud ai-platform jobs submit training ${JOB_NAME_4} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_PIPELINE_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME_Face} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--scale-tier=CUSTOM \
		--master-machine-type=n1-highmem-8 \
		--stream-logs
