
# coding: utf-8

# #Modules

# In[3]:

from elasticsearch import Elasticsearch
from py2neo import Graph, Node, Relationship


# #ElasticSearch

# In[4]:

#the index client used to communicate with the database
client = Elasticsearch()
#the index name
indexName = "gastronomical"
docType = 'recipes'


# #Neo4J

# In[5]:

# Graph database entity
graph_db = Graph("http://neo4j:neo4ja@localhost:7474/db/data/", username='neo4j', password='neo4j')


# #Ingredients data

# In[1]:

# ingredients
filename = './ingredients.txt'
ingredients =[]
with open(filename) as f:
    for line in f:
        # strip because of the /n you get otherwise from reading the .txt
        ingredients.append(line.strip())

print(ingredients)


# #ElasticSearch to Neo4J 

# In[ ]:

ingredientnumber = 0
grandtotal = 0
for ingredient in ingredients:
    try:
        a = Node("Ingredient", Name=ingredient)
        IngredientNode = graph_db.merge(a,"Ingredient","Name")
    except:
        continue

    ingredientnumber +=1
    searchbody = {
        "size" : 10000,
        "query": {
            "match_phrase":
                {
                    "ingredients":{
                        "query":ingredient,
                    }
                }
        }
    }

    result = client.search(index=indexName,body=searchbody)

    print(ingredient)
    print(ingredientnumber)
    print("total: " +  str(result['hits']['total']))

    grandtotal = grandtotal + result['hits']['total']['value']
    print("grand total: " +  str(grandtotal))

    for recipe in result['hits']['hits']:

        try:
            b = Node("Recipe", Name=recipe['_source']['name'])
            RecipeNode = graph_db.merge(b, "Recipe", "Name")
            #RecipeNode = graph_db.merge_one("Recipe","Name",recipe['_source']['name'])
            NodesRelationship = Relationship(b, "Contains", a)

            graph_db.create(NodesRelationship)
            print("added: " + recipe['_source']['name'] + " contains " + ingredient)

        except:
            continue

    print("*************************************")

