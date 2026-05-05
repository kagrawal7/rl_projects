python3 -m venv .venv
chmod u+x .venv/bin/activate
source .venv/bin/activate
pip install -r requirements.txt
python3 -m ipykernel install --user --name=aps1080_a2_jupyter --display-name "aps1080_a2_jupyter"
cp -r "$LATEX_NBCONVERT_TEMPLATE" .venv/share/jupyter/nbconvert/templates
code .
