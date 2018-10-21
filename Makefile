setup:
	#You may want to create an alias to automatically source this:
	# alias functop="cd ~/src/functional_intro_to_python && source ~/.func/bin/activate"
	python3 -m venv ~/.func

install:
	pip install -r requirements.txt

test:
	#PYTHONPATH=. && pytest -vv --cov=paws --cov=spot-price-ml tests/*.py
	PYTHONPATH=. && py.test --nbval-lax Assignment_01/*.ipynb
	PYTHONPATH=. && py.test --nbval-lax Assignment_02/*.ipynb
	PYTHONPATH=. && py.test --nbval-lax Assignment_03/*.ipynb
	
lint:
	pylint --disable=R,C funclib

convert:
	jupyter nbconvert --to pdf Assignment_01/Assignment_02.ipynb --post serve
  jupyter nbconvert --to pdf Assignment_02/Assignment_02.ipynb --post serve
  jupyter nbconvert --to pdf Assignment_02/Assignment_02.ipynb --post serve
        
all: install lint test convert