import matplotlib.pyplot as plt
import dhg
# draw a graph
#g = dhg.random.graph_Gnm(10, 12)
#g.draw()
# draw a hypergraph
hg = dhg.random.hypergraph_Gnm(10, 8)
hg.draw()
# show figures
plt.show()