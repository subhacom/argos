:: Run the argos app
@echo on
pushd "%~dp0"
echo "Current directory"
set PYTHONPATH=.;$PYTHONPATH
python argos\amain.py
popd