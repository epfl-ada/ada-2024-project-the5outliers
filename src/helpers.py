def create_treemap_data(df):
    """
    Processes the DataFrame Categories to generate labels, parents, values, and ids for the treemap.
    """
    labels = []
    ids = []
    parents = []
    values = []

    # Create a dictionary to keep track of node IDs
    node_ids = {}

    # Get counts at each level
    counts_level_1 = df.groupby('level_1').size().reset_index(name='count')
    counts_level_2 = df.groupby(['level_1', 'level_2']).size().reset_index(name='count')
    counts_level_3 = df.groupby(['level_1', 'level_2', 'level_3']).size().reset_index(name='count')

    # Process level 1 nodes
    for _, row in counts_level_1.iterrows():
        level_1 = row['level_1']
        label = level_1
        id = level_1
        parent_id = ''
        labels.append(label)
        ids.append(id)
        parents.append(parent_id)
        values.append(row['count'])
        node_ids[(level_1,)] = id

    # Process level 2 nodes
    for _, row in counts_level_2.iterrows():
        level_1 = row['level_1']
        level_2 = row['level_2']
        label = level_2
        id = f"{level_1}/{level_2}"
        parent_id = level_1
        labels.append(label)
        ids.append(id)
        parents.append(parent_id)
        values.append(row['count'])
        node_ids[(level_1, level_2)] = id

    # Process level 3 nodes
    for _, row in counts_level_3.iterrows():
        level_1 = row['level_1']
        level_2 = row['level_2']
        level_3 = row['level_3']
        label = level_3
        id = f"{level_1}/{level_2}/{level_3}"
        parent_id = f"{level_1}/{level_2}"
        labels.append(label)
        ids.append(id)
        parents.append(parent_id)
        values.append(row['count'])
        node_ids[(level_1, level_2, level_3)] = id

    return labels, parents, values, ids