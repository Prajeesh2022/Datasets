import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
import networkx as nx 
import matplotlib.pyplot as plt 
import numpy as np 
from torch.nn import Linear 
from torch_geometric.nn import GCNConv
dataset = KarateClub()
data = dataset[0]
print(dataset.num_features)
import pandas as pd
node_features_df = pd.DataFrame(data.x.numpy())
node_features_df['node'] = node_features_df.index
node_features_df.set_index('node',inplace = True)
edge_index_df = pd.DataFrame(data.edge_index.numpy().T)
labels_df = pd.DataFrame(data.y.numpy(), columns = ['labels'])
labels_df['node'] = labels_df.index
labels_df.set_index('node',inplace = True)
#print(node_features_df)
num_edges = data.edge_index.shape[1]
print(num_edges)
print(edge_index_df)
print(labels_df.head())
num_countries = 4
np.random.seed(42)
countries = torch.tensor(np.random.choice(num_countries,data.num_nodes))
data.y = countries
#print(data.x)
#print(data.edge_index)
#
# print(data.y)
# Graph plotting
#G = to_networkx(data,to_undirected=True)
#plt.figure(figsize = (12,12))
#plt.axis("off")
#nx.draw_networkx(G,pos = nx.spring_layout(G,seed =0),with_labels = True, node_size = 800,node_color = data.y, cmap="hsv",vmin = -2,vmax = 3,width = 0.8, edge_color = "grey", font_size= 14)
#plt.show()
#plt.savefig("graph.png")

# neural network model
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = GCNConv(dataset.num_features,3)
        self.out = Linear(3, num_countries)
    def forward(self,x,edge_index):
        h= self.gcn(x,edge_index).relu()
        z = self.out(h)
        return h,z
model = GCN()
print(model)
