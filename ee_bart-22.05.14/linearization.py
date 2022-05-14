import abc
from typing import List, Optional, Dict, Any, Set, TypeVar

from cached_property import cached_property
from dataclasses import dataclass

entity_dict = {'VEH': 'Vehicle', 'PER': 'Person', 'LOC': 'Location', 'Job-Title': 'Job-Title',
               'ORG': 'Organization',
               'GPE': 'Geopolitical-Entity', 'Time': 'Time', 'FAC': 'Facility', 'Numeric': 'Numeric',
               'WEA': 'Weapons', 'TIM': 'Time',
               'Sentence': 'Sentence', 'Crime': 'Crime', 'Contact-Info': 'Contact-Information'}


@dataclass
class EventGraph:
    nodes: List[str]
    """
    List of linearized nodes, with special tokens.
    """

    ee_graph_dict: Dict[str, Any]
    """
    Dict of Event Graph
    """

    @cached_property
    def ee_graph_dict(self) -> Set[str]:
        """Dict of variables in this event graph"""
        return self.ee_graph_dict

    @cached_property
    def backreferences(self):
        return [i for i in range(len(self.nodes))]

    @cached_property
    def nodes(self) -> List[str]:
        """Linearized nodes with event_type,relation,argument"""
        return self.nodes

    def src_occurrence(self, var: str) -> int:
        pass


class BaseLinearizer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def linearize(self, *args, **kwargs) -> EventGraph:
        pass


class ACETokens:
    START, END = '<', '>'
    _TEMPL = START + '{}' + END
    BOS_N = _TEMPL.format('s')
    EOS_N = _TEMPL.format('/s')
    START_N = _TEMPL.format('start')
    STOP_N = _TEMPL.format('stop')
    EVENT_N = _TEMPL.format('Event')
    ARGUMENT_N = _TEMPL.format('Argument')
    Yes_N = _TEMPL.format('yes')
    No_N = _TEMPL.format('no')
    ENTITY_N = _TEMPL.format('entity')
    BOS_E = _TEMPL.format('s')
    EOS_E = _TEMPL.format('/s')
    START_E = _TEMPL.format('start')
    STOP_E = _TEMPL.format('stop')

    _FIXED_SPECIAL_TOKENS_N = {
        BOS_N, EOS_N, START_N, STOP_N}
    _FIXED_SPECIAL_TOKENS_E = {
        BOS_E, EOS_E, START_E, STOP_E}
    _FIXED_SPECIAL_TOKENS = _FIXED_SPECIAL_TOKENS_N | _FIXED_SPECIAL_TOKENS_E

    @classmethod
    def is_node(cls, string: str) -> bool:
        if isinstance(string, str) and string.startswith(':'):
            return False
        elif string in cls._FIXED_SPECIAL_TOKENS_E:
            return False
        return True


T = TypeVar('T')


def add_seq_label(node_list):
    node_seq_start = ACETokens.BOS_N
    node_seq_end = ACETokens.EOS_N
    node_list.insert(0, node_seq_start)
    node_list.append(node_seq_end)
    return node_list


class ACELinearizer(BaseLinearizer):

    def __init__(
            self,
            use_pointer_tokens: bool = True,
            collapse_name_ops: bool = False,
            dfs_linearization: bool = True,
            bfs_linearization: bool = True,
            use_entity_type: bool = True,
            use_classification: bool = True
    ):

        self.interleave_edges = False
        self.use_pointer_tokens = use_pointer_tokens
        self.dfs_linearization = dfs_linearization
        self.bfs_linearization = bfs_linearization
        self.use_entity_type = use_entity_type
        self.use_classification = use_classification

    def linearize(self, event_mentions) -> EventGraph:
        # 得到event_graph
        linearized = self._linearize(event_mentions)  # 得到event_type+role+argument
        # 得到event_sequence
        if linearized != [ACETokens.No_N]:
            linearized1 = self._interleave(linearized)
        else:
            linearized1 = linearized
        if self.use_pointer_tokens:
            if linearized1 != [ACETokens.No_N]:
                linearized1 = self._add_special_ee_tokens(linearized1)
            else:
                linearized1 = add_seq_label([ACETokens.No_N])
        return linearized1, linearized

    # get event graph
    def _linearize(self, event_mentions) -> EventGraph:
        event_graph = []
        for i in range(len(event_mentions)):
            event_dict = {}
            event_dict['event_type'] = event_mentions[i]['event_type']
            arguments = event_mentions[i]['arguments']
            arg_list = []
            for arg in arguments:
                # get role
                arg_dict = {}
                arg_dict['relation'] = arg['role']
                if self.use_entity_type:
                    arg_dict['entity_type'] = arg['entity-type']
                head = arg['head'].split()
                text = arg['text'].split()
                if head[-1] in text:
                    arg_dict['argument'] = head[-1]
                else:
                    if len(text) == 1:
                        if head[-1] in arg['text'].split('-'):
                            arg_dict['argument'] = head[-1]
                        else:
                            arg_dict['argument'] = text[-1]
                    else:
                        arg_dict['argument'] = text[-1]
                arg_list.append(arg_dict)
            event_dict['argument'] = arg_list
            event_graph.append(event_dict)
        if self.use_classification:
            if event_graph == []:
                event_graph.append(ACETokens.No_N)
        return event_graph

    # get event sequence
    def _interleave(self, event_graph) -> EventGraph:

        if self.dfs_linearization:
            new_nodes = []
            if self.use_classification:
                new_nodes.append(ACETokens.Yes_N)
            for i in range(len(event_graph)):
                new_nodes.append(event_graph[i]['event_type'])
                for node in event_graph[i]['argument']:
                    if self.use_entity_type:
                        new_nodes.append(entity_dict[node['entity_type'].split(':')[0]])
                    new_nodes.append(node['argument'])
                    new_nodes.append(node['relation'])
                new_nodes.append(ACETokens.STOP_N)
            # add_seq_label
            new_nodes = add_seq_label(new_nodes)
            new_event_graph = EventGraph(new_nodes, event_graph)
            return new_event_graph
        if self.bfs_linearization:
            new_nodes = []
            for i in range(len(event_graph)):
                new_nodes.append(event_graph[i]['event_type'])
            # new_nodes.append(ACETokens.STOP_N)
            for i in range(len(event_graph)):
                for node in event_graph[i]['argument']:
                    if self.use_entity_type:
                        new_nodes.append(entity_dict[node['entity_type'].split(':')[0]])
                    new_nodes.append(node['argument'])
                    new_nodes.append(node['relation'])
                # 添加<stop>结点
                # if len(event_graph[i]['argument']) > 0:
                new_nodes.append(ACETokens.STOP_N)
            # add_seq_label
            new_nodes = add_seq_label(new_nodes)
            new_event_graph = EventGraph(new_nodes, event_graph)
            return new_event_graph

    # add special label
    def _add_special_ee_tokens(self, graph: EventGraph) -> EventGraph:
        if self.dfs_linearization:
            new_graph = []
            if self.use_classification:
                new_graph.append(ACETokens.Yes_N)
            event_dict = graph.ee_graph_dict
            id = 0
            for i in range(len(event_dict)):
                # new_graph.append(f"<Event:{i}>")
                new_graph.append(event_dict[i]['event_type'])
                for node in event_dict[i]['argument']:
                    if self.use_entity_type:
                        new_graph.append(entity_dict[node['entity_type'].split(':')[0]])
                    if node['argument'] not in new_graph:
                        new_graph.append(f"<Argument:{id}>")
                        new_graph.append(node['argument'])
                        id += 1
                    else:
                        index = new_graph.index(node['argument'])
                        new_graph.append(new_graph[index - 1])
                    new_graph.append(node['relation'])
                new_graph.append(ACETokens.STOP_N)
            # add_seq_label
            new_graph = add_seq_label(new_graph)
            new_event_graph = EventGraph(new_graph, event_dict)
            return new_event_graph
        if self.bfs_linearization:
            new_graph = []
            event_dict = graph.ee_graph_dict
            # 添加<Event>节点
            for i in range(len(event_dict)):
                # new_graph.append(f"<Event:{i}>")
                new_graph.append(event_dict[i]['event_type'])
            # new_graph.append(ACETokens.STOP_N)
            id = 0
            for i in range(len(event_dict)):
                # if len(event_dict[i]['argument']) > 0:
                #     index = new_graph.index(event_dict[i]['event_type'])
                #     new_graph.append(new_graph[index - 1])
                for node in event_dict[i]['argument']:
                    if self.use_entity_type:
                        new_graph.append(entity_dict[node['entity_type'].split(':')[0]])
                    if node['argument'] not in new_graph:
                        new_graph.append(f"<Argument:{id}>")
                        new_graph.append(node['argument'])
                        id += 1
                    else:
                        index = new_graph.index(node['argument'])
                        new_graph.append(new_graph[index - 1])
                    new_graph.append(node['relation'])
                # 添加<stop>结点
                new_graph.append(ACETokens.STOP_N)
            # add_seq_label
            new_graph = add_seq_label(new_graph)
            new_event_graph = EventGraph(new_graph, event_dict)
            return new_event_graph
