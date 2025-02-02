from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, Collection
from utils import embedding_fn

client = MilvusClient(
    uri='',
    token=""
)

mcu_dcu_texts = [
    "James Gunn's 'Superman: Legacy' aims to redefine the Man of Steel with a fresh yet classic approach, kicking off a new DCU era.",
    "The MCU's 'Avengers: Secret Wars' is expected to be the biggest crossover event in Marvel history, potentially resetting the entire multiverse.",
    "Matt Reeves' 'The Batman Part II' will continue Robert Pattinson's noir-driven take on the Dark Knight, diving deeper into Gotham’s corruption.",
    "With 'The Brave and the Bold,' DCU introduces a new Batman and Robin dynamic, focusing on the father-son relationship between Bruce and Damian Wayne.",
    "Marvel's 'Fantastic Four' reboot is set to bring the iconic superhero family into the MCU, finally integrating them into the larger Marvel storyline.",
    "James Gunn and Peter Safran are crafting a connected DCU where TV, movies, and animation share a single, cohesive narrative.",
    "Sam Raimi’s return to Marvel with 'Doctor Strange in the Multiverse of Madness' brought horror elements into the MCU, teasing darker storylines.",
    "DCU’s 'Supergirl: Woman of Tomorrow' is inspired by Tom King’s comic run, portraying Kara Zor-El as a fierce warrior on an interstellar journey.",
    "The upcoming 'Kang Dynasty' will solidify Kang the Conqueror as the MCU's next big threat, spanning multiple timelines and realities.",
    "James Gunn's 'Creature Commandos' will be the first animated project in the new DCU, proving that every medium can be interconnected.",
    "Marvel Studios’ 'X-Men' reboot is one of the most anticipated MCU projects, as fans eagerly await the mutant integration into the universe.",
    "DCU’s 'Lanterns' series will focus on Hal Jordan and John Stewart as space cops investigating a larger cosmic mystery tied to the main DCU storyline.",
    "'Deadpool 3' will officially bring Wade Wilson into the MCU, featuring a rumored multiversal road trip with Wolverine.",
    "The upcoming 'Batman Beyond' project in the DCU could explore Terry McGinnis' future Gotham under the mentorship of an aging Bruce Wayne.",
    "Marvel’s 'Thunderbolts' will assemble antiheroes and reformed villains in a mission-based film that could shake up the MCU's status quo.",
    "James Gunn has confirmed that the DCU's version of Superman will embody hope, kindness, and optimism while still being deeply relatable.",
    "Marvel’s 'Blade' reboot starring Mahershala Ali is set to introduce vampires and supernatural elements into the MCU’s growing dark universe.",
    "DCU’s 'The Authority' will introduce morally complex heroes operating in a gray area, a stark contrast to traditional superheroes.",
    "'Spider-Man 4' in the MCU is rumored to explore a more grounded, street-level Peter Parker dealing with the consequences of 'No Way Home.'",
    "James Gunn’s DCU will have a carefully planned 10-year roadmap, ensuring consistent storytelling across films, shows, and games."
]

def generate_random_data(num_docs=20):
    docs = []
    for i in range(num_docs):
        text = mcu_dcu_texts[i]
        embeddings = embedding_fn(text)  # Generate embeddings
        docs.append({"text": text, "embedding": embeddings})
    return docs

def insert():
    collection_name = "marveldcc"

    # Define schema correctly
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        ],
        description="Collection for random documents"
    )

    # Create the collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema
    )
    docs = []
    for i in range(20):
        text = mcu_dcu_texts[i]
        print(text)
        embeddings = embedding_fn(text)  # Generate embeddings
        docs.append({"text": text, "embedding": embeddings})

    print(docs)
    for i in range(20):
        insert_data = {
            "text": docs[i]["text"] ,
            "embedding": docs[i]["embedding"]
        }
        client.insert(collection_name=collection_name, data=insert_data)

    print(f"Inserted {len(docs)} random documents into Milvus collection '{collection_name}'")

def get_docs():
    
    collection_name = "marveldcc"
    client.load_collection(collection_name)

    # Querying the collection
    query_vectors = embedding_fn(["DCU Authority"])
    
    res = client.search(
        collection_name=collection_name, 
        data=query_vectors,  
        limit=1, 
        output_fields=["text"],
    )
    
    docs = [doc["entity"]["text"] for doc in res[0]]
    return docs

def create_index():
    from pymilvus import Collection, connections

    connections.connect(
        alias="default", 
        uri=""
    )

    collection_name = "marveldcc"

    collection = Collection(collection_name)

    index_params = {
        "index_type": "IVF_FLAT",  # Index type, can also use IVF_SQ8 or HNSW
        "metric_type": "L2",        # Distance metric, e.g., L2 or IP
        "params": {"nlist": 384}  # Number of clusters
    }


    collection.create_index(field_name="embedding", index_params=index_params)


    print("Index created successfully!")


collections = client.list_collections()
print(f"Number of collections in the database: {len(collections)}")
print("collections : ", collections)
insert()
create_index()
print(get_docs())
