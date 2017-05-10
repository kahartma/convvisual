export SUBJECT=$SGE_TASK_ID

cd /home/hartmank/braindecode
source vienv/bin/activate
cd convvisual/notebooks
runipy Paper_Make_All_Figures_From_Data.ipynb out_notebooks/${SGE_TASK_ID}Paper_Make_All_Figures_From_Data.ipynb