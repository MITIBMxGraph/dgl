import dgl
from dgl.base import DGLError
#from dgl import backend as F
import torch
import pytest

test_type = torch.int32

def create_test_heterograph(idtype):
    # test heterograph from the docstring, plus a user -- wishes -- game relation
    # 3 users, 2 games, 2 developers
    # metagraph:
    #    ('user', 'follows', 'user'),
    #    ('user', 'plays', 'game'),
    #    ('user', 'wishes', 'game'),
    #    ('developer', 'develops', 'game')])

    g = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 2, 1], [0, 0, 1, 1]),
        ('user', 'wishes', 'game'): ([0, 2], [1, 0]),
        ('developer', 'develops', 'game'): ([0, 1], [0, 1])
    }, idtype=idtype)
    assert g.idtype == idtype
    assert g.device.type == 'cpu'
    return g

def test_pin_memory_(idtype):
    # TODO: rewrite this test case to accept different graphs so we
    #  can test reverse graph and batched graph
    g = create_test_heterograph(idtype)
    g.nodes['user'].data['h'] = torch.ones((3, 5), dtype=test_type)
    g.nodes['game'].data['i'] = torch.ones((2, 5), dtype=test_type)
    g.edges['plays'].data['e'] = torch.ones((4, 4), dtype=test_type)

    g.nodes['user'].data['h'].device.type == 'cpu'
    g.nodes['game'].data['i'].device.type == 'cpu'
    g.edges['plays'].data['e'].device.type == 'cpu'
    assert not g.is_pinned()

    # unpin an unpinned CPU graph, directly return
    g.unpin_memory_()
    assert not g.is_pinned()
    assert g.device.type == 'cpu'

    # pin a CPU graph
    g.pin_memory_()
    assert g.is_pinned()
    assert g.device.type == 'cpu'

    assert g.nodes['user'].data['h'].device.type == 'cpu'
    assert g.nodes['game'].data['i'].device.type == 'cpu'
    assert g.edges['plays'].data['e'].device.type == 'cpu'
    for ntype in g.ntypes:
        assert g.batch_num_nodes(ntype).device.type == 'cpu'
    for etype in g.canonical_etypes:
        assert g.batch_num_edges(etype).device.type == 'cpu'

    # not allowed to create new formats for the pinned graph
    with pytest.raises(DGLError):
        g.create_formats_()
    # it's fine to clone with new formats, but new graphs are not pinned
    # >>> g.formats()
    # {'created': ['coo'], 'not created': ['csr', 'csc']}
    assert not g.formats('csc').is_pinned()
    assert not g.formats('csr').is_pinned()
    # 'coo' formats is already created and thus not cloned
    assert g.formats('coo').is_pinned()

    # pin a pinned graph, direcly return
    g.pin_memory_()
    assert g.is_pinned()
    assert g.device.type == 'cpu'

    # unpin a pinned graph
    g.unpin_memory_()
    assert not g.is_pinned()
    assert g.device.type == 'cpu'

    device = torch.device('cuda:0')
    g1 = g.to(device)

    # unpin an unpinned GPU graph, directly return
    g1.unpin_memory_()
    assert not g1.is_pinned()
    assert g1.device.type == 'cuda'


    assert g1.nodes['user'].data['h'].device.type == 'cuda'
    assert g1.nodes['game'].data['i'].device.type == 'cuda'
    assert g1.edges['plays'].data['e'].device.type == 'cuda'
    for ntype in g1.ntypes:
        assert g1.batch_num_nodes(ntype).device.type == 'cuda'
    for etype in g1.canonical_etypes:
        assert g1.batch_num_edges(etype).device.type == 'cuda'

    # error pinning a GPU graph
    with pytest.raises(DGLError):
        g1.pin_memory_()

test_pin_memory_(test_type)
