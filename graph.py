class Vertex(object):
    def __init__(self, v_id, vlb):
        self.v_id = v_id
        self.vlb = vlb
        self.adjacency_dic = {}
        

    def add_adjacent_vertex(self, to, tolb, eid):
        if tolb not in self.adjacency_dic:                        
            self.adjacency_dic[tolb] = []
        self.adjacency_dic[tolb].append(to)


class Graph(object):
    def __init__(self, g_id):
        self.g_id = g_id
        self.vertices = {} # dictionary of vertices(key: verted id, value: a Vartex class object)
        self.edges = set() # set of edge indices
        self.search_dic = {}  # (key: vertex label, list of vertices that have key's label)
        self.cache = {} # (key: graph operation, value: corresponding vertices)
        self.edge_info_dic = {} # {edge id: (v1, v2, elb)}
        
    
    def add_vertex(self, v_id, vlb):
        self.vertices[v_id] = Vertex(v_id, vlb)
    
    def add_edge(self, eid, v1, v2, elb=0):
        self.edges.add(eid)
        self.edge_info_dic[eid] = (v1, v2, elb)
    
    def return_vertices(self): #for debug
        for vertex in self.vertices.values():
            print(vertex.v_id, vertex.vlb)
        
    
    def make_adjacency_list(self, frm, to, eid, elb=0):
        if frm not in self.vertices or to not in self.vertices:
            print("an error occured")
        self.vertices[frm].add_adjacent_vertex(to, self.vertices[to].vlb, eid)
        
        q1 = (frm, self.vertices[to].vlb)
        q2 = (frm, to, elb)
        if q1 not in self.cache: self.cache[q1] = []
        self.cache[q1].append((to, eid))
        if q2 not in self.cache: self.cache[q2] = []
        self.cache[q2].append(eid)
    
    def make_search_dic(self):
        for vertex in self.vertices.values():
            if vertex.vlb not in self.search_dic: self.search_dic[vertex.vlb] = []
            self.search_dic[vertex.vlb].append(vertex.v_id)
    
    def fordebug(self):
        print("hello")
        for v in self.vertices.values():
            print(v.adjacency_edge_set)
        exit()
