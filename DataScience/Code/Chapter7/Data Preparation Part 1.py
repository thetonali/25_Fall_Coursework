
# coding: utf-8

# #Modules

# In[1]:

from elasticsearch import Elasticsearch
import json


# #ElasticSearch

# ##Index

# In[ ]:

#the index name Elasticsearch client used to communicate with the database
client = Elasticsearch()
indexName = "gastronomical"
docType = 'recipes'


# In[ ]:

# create an index (only once)
#client.indices.create(index=indexName)


# ##Document

# In[ ]:

# location of recipe json file: change this to match your own setup!
file_name = './recipes.json'


# In[ ]:

#Create document mapping
recipeMapping = {
        'properties': {
            'name': {'type': 'text'},
            'ingredients': {'type': 'text'}
        }
    }
client.indices.put_mapping(index=indexName,doc_type=docType,body=recipeMapping,include_type_name=True)


# #Recipes Data

# In[ ]:

#Load Json file
with open(file_name, encoding='utf-8') as data_file:
    recipeData = json.load(data_file)


# #Indexing

# In[ ]:

#Index the recipes
for recipe in recipeData:
    print(recipe.keys())
    print(recipe['_id'].keys())
    client.index(index=indexName, doc_type=docType,id = recipe['_id']['$oid'], body={"name": recipe['name'], "ingredients":recipe['ingredients']})

