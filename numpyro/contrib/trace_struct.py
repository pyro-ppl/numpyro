from collections import OrderedDict


class TraceStructure:
    """
    Graph structure denoting the relationship among pyro primitives in the execution path.
    """

    def __init__(self):
        self.nodes = OrderedDict()
        self._successors = OrderedDict()
        self._predecessors = OrderedDict()

    def __contains__(self, site):
        return site in self.nodes

    def add_edge(self, from_site, to_site):
        for site in (from_site, to_site):
            if site not in self:
                self.add_node(site)

        self._successors[from_site].add(to_site)
        self._predecessors[to_site].add(to_site)

    def add_node(self, site_name, **kwargs):
        if site_name in self:
            # TODO: handle reused name!
            pass
        self.nodes[site_name] = kwargs
        self._successors[site_name] = set()
        self.__predecessors[site_name] = set()

    def predecessor(self, site):
        return self._predecessors[site]

    def successor(self, site):
        return self._successors[site]

    # TODO: remove edge
