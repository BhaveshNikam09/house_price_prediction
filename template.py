import os
from pathlib import Path


package_name='HousePricePrediction'

list_of_files=[
    ".github/workflows/.gitkeep",
    f'src/{package_name}/__init__.py',
    f'src/{package_name}/components/__init__.py',
    f'src/{package_name}/components/data_ingestion.py',
    f'src/{package_name}/components/data_transformation.py',
    f'src/{package_name}/components/model_trainer.py',
    f'src/{package_name}/pipelines/__init__.py',
    f'src/{package_name}/pipelines/training_pipeline.py',
    f'src/{package_name}/pipelines/prediction_pipeline.py',
    f'src/{package_name}/logger.py',
    f'src/{package_name}/exception.py',
    f'src/{package_name}/utils/__init__.py',
    'notebooks/research.ipynb',
    'notebooks/data/.gitkeep',
    'requirements.txt',
    'setup.py',
    'init_setup.sh',
]


# here we will create directory

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)
    
    
    if filedir!='':
        os.makedirs(filedir,exist_ok=True)
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,"w") as f :
            pass
        
    else:
        print(f"{filename} already exists")
        
## this is the basic file handling 