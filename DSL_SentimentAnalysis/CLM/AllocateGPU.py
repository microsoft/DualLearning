import sys

mapper_machine_freecard = {}
mapper_machine_rank = {}

def MapIDs(m_machine):
    for i in range(m_machine):
        fo = open('record' + str(i))
        id = 0
        m_line = 0
        machine_name = ''
        for line in fo:
            if id == 0:
                machine_name = line[:-1]
                mapper_machine_freecard[machine_name] = []
                if mapper_machine_rank.has_key(machine_name):
                    mapper_machine_rank[machine_name].append(i)
                else:
                    mapper_machine_rank[machine_name] = [i]
            elif id > 1:
                mapper_machine_freecard[machine_name].append(int(line))
            id = id + 1
        fo.close()

def Map_Rank_Card(m_machine):
    MapIDs(m_machine)
    allocations = range(m_machine)
    for k in mapper_machine_rank.keys():
        ranks = mapper_machine_rank[k]
        cards = mapper_machine_freecard[k]
    #if len(ranks) == len(cards):
        for i in range(len(ranks)):
            allocations[ranks[i]] = cards[i]
    
    for l in allocations:
        print l


if __name__ == '__main__':
    Map_Rank_Card(int(sys.argv[1]))
