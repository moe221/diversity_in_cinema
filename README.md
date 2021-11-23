# Data analysis
- Document here the project: diversity_in_cinema
- Description: Project Description
- Data Source:
- Type of analysis:

Please document the project the better you can.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for diversity_in_cinema in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/diversity_in_cinema`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "diversity_in_cinema"
git remote add origin git@github.com:{group}/diversity_in_cinema.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
diversity_in_cinema-run
```

# Install

Go to `https://github.com/{group}/diversity_in_cinema` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/diversity_in_cinema.git
cd diversity_in_cinema
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
diversity_in_cinema-run
```

# How to contribute
- Always create a separate branch for new features 

    ```bash
    git branch [FEATURE_NAME]
    git checkout [FEATURE_NAME]
    ```
- Update the requirements.txt every time you use a new library
- Add and commit changes as soon as you mak them
- Create pull requests before merging your branch
- Comment your code and add function docstrings 
