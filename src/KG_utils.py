
def triplet_initializer(dataset_file):
    file = open(dataset_file, "r")
    id_entity = 0
    id_relation = 0
    
    entities_to_ids = {}
    relations_to_ids = {}
    
    triplets = []
    
    for line in file:
        
        
        ent1, rel, ent2 = line.split(sep="\t")
        
        if ent1 not in entities_to_ids:
            entities_to_ids[ent1] = id_entity
            id_entity += 1
        
        if rel not in relations_to_ids:
            relations_to_ids[rel] = id_relation
            id_relation += 1
            
        if ent2 not in entities_to_ids:
            entities_to_ids[ent2] = id_entity
            id_entity += 1
        
        triplet = [entities_to_ids[ent1], relations_to_ids[rel], entities_to_ids[ent2]]    
        triplets.append(triplet)
        
    ids_to_entities = {v:k for k,v in entities_to_ids.items()}
    ids_to_relation = {v:k for k,v in relations_to_ids.items()}
    return entities_to_ids, relations_to_ids, ids_to_entities, ids_to_relation, triplets