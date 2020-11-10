
## Video Sentiment Analysis

Repository for the Video sentiment analysis module of the ESA project.

## Setup the project

1. Clone/pull project
* If it is from scratch, clone this project:
```
git clone https://ooti-projects.win.tue.nl/gitlab/st-c2019/esa/video-sentimental-analysis.git
```
* if it is from existing project,  pull the project 
```
git pull
```

2. Follow the instructions to install DVC (on Windows): 
* Linux: https://dvc.org/doc/install/linux
* MacOS: https://dvc.org/doc/install/macos
* Windows: https://dvc.org/doc/install/windows 

3. Follow the instructions to install aws cli version 2.
* Linux: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html
* MacOS: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-mac.html
* Windows: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-windows.html

4. Configure aws cli:
```
> aws configure
AWS Access Key ID [None]: AKIA55GEKVQR3FCPU7PV
AWS Secret Access Key [None]: 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
Default region name [None]: eu-central-1
Default output format [None]: text
```

5. Download production data and models:
```
dvc pull
```
## Use cases

### Modify production data
1. Add or remove audios from the ```prod_data``` directory.
2. Execute ```dvc status``` to see that the contents of ```prod_data``` were modified.
3. Execute ```dvc add prod_data``` to update the contents of ```prod_data.dvc```.
4. Track changes with git: ```git add prod_data.dvc```.
5. Git commit: ```git commit -m "commit message" ```.
6. Update production: ```dvc push```.
7. Push changes to GitLab: ```git push origin [branch-name]```.

### Add new production model
1. Add new model in the ```prod_models``` directory.
2. Execute ```dvc status``` to see that the contents of ```prod_models``` were modified.
3. Execute ```dvc add prod_models``` to update the contents of ```prod_models.dvc```.
4. Track changes with git: ```git add prod_models.dvc```.
5. Git commit: ```git commit -m "commit message" ```.
6. Update production: ```dvc push```.
7. Push changes to GitLab: ```git push origin [branch-name]```.


### Deploy new model in the raspberry pi
1. Log into the raspberry pi.
2. Go to **/home/pi/esaProject**.
3. Activate virtual environment: ``` source videoTeam/bin/activate ```.
4. Go to the repository directory```cd video-sentiment-analysis```.
5. Pull changes from master ```git pull origin master```.
6. Pull latest data and models using dvc ```dvc pull```.
6. Go to the repository directory containing the source code.
7. Run your script.