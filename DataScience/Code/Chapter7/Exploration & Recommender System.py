
# coding: utf-8

# #Modules

# In[2]:

from py2neo import Graph, Node, Relationship


# #Database

# In[3]:

graph_db = Graph("http://neo4j:neo4ja@localhost:7474/db/data/")


# #Exploration

# ##Top 10 Ingredients

# In[18]:

graph_db.cypher.execute("MATCH (REC:Recipe)-[r:Contains]->(ING:Ingredient) WITH ING, count(r) AS num RETURN ING.Name as Name, num ORDER BY num DESC LIMIT 10;")


# ##Top 10 Recipes with the biggest number of ingredients

# In[14]:

graph_db.cypher.execute("MATCH (REC:Recipe)-[r:Contains]->(ING:Ingredient) WITH REC, count(r) AS num RETURN REC.Name as Name, num ORDER BY num DESC LIMIT 10;")


# ##Spaghetti Bolognese

# In[15]:

graph_db.cypher.execute("MATCH (REC1:Recipe{Name:'Spaghetti Bolognese'})-[r:Contains]->(ING:Ingredient) RETURN REC1.Name, ING.Name;")


# #Recommendation

# ##Add User

# In[4]:

UserNode = graph_db.merge_one("User","Name","Ragnar")


# ##Add User likes

# In[5]:

UserRef = graph_db.find_one("User",property_key="Name", property_value="Ragnar")#look for user Ragnar


# In[6]:

RecipeRef = graph_db.find_one("Recipe",property_key="Name", property_value="Spaghetti Bolognese") #look for recipe Spaghetti Bolognese
NodesRelationship = Relationship(UserRef, "Likes", RecipeRef) #Ragnar likes Spaghetti Bolognese
graph_db.create_unique(NodesRelationship) #Commit his like to database


# In[7]:

graph_db.create_unique(Relationship(UserRef, "Likes", graph_db.find_one("Recipe",property_key="Name", property_value="Roasted Tomato Soup with Tiny Meatballs and Rice")))
graph_db.create_unique(Relationship(UserRef, "Likes", graph_db.find_one("Recipe",property_key="Name", property_value="Moussaka")))
graph_db.create_unique(Relationship(UserRef, "Likes", graph_db.find_one("Recipe",property_key="Name", property_value="Chipolata &amp; spring onion frittata")))
graph_db.create_unique(Relationship(UserRef, "Likes", graph_db.find_one("Recipe",property_key="Name", property_value="Meatballs In Tomato Sauce")))
graph_db.create_unique(Relationship(UserRef, "Likes", graph_db.find_one("Recipe",property_key="Name", property_value="Macaroni cheese")))
graph_db.create_unique(Relationship(UserRef, "Likes", graph_db.find_one("Recipe",property_key="Name", property_value="Peppered Steak")))


# ##Recommend recipes to User

# In[19]:

graph_db.cypher.execute("MATCH (USR1:User{Name:'Ragnar'})-[l1:Likes]->(REC1:Recipe),(REC1)-[c1:Contains]->(ING1:Ingredient) WITH  ING1,REC1 MATCH (REC2:Recipe)-[c2:Contains]->(ING1:Ingredient) WHERE REC1 <> REC2 RETURN REC2.Name,count(ING1) AS IngCount ORDER BY IngCount DESC LIMIT 20;")


# In[ ]:



