# to-do list:

\- finish testing inference.py 

&#x09;- change this if things go wrong idk

\- probably look into different models as mentioned

\- try to look into merging the pkl files so its on pkl for a model to be trained on

&#x09;- i was trying this and nothing really worked



### commands to use on putty

**to process multiple categories:**

*python data/run\_multicategories.py --gpu --config categories.json > full\_run\_production.log*
- pretty sure you don't need the production log because the script should just spit out a directory with logs anyways

\- make sure to change the categories.json whenever you want to add new datasets



**to run the model:**

*python models/new\_model.py -- path output/dev\_(category name).parquet*

