
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
import cv2
import sys

# Gaurvi path
# sys.path.append('/home/code/hpe-core/')

# path to lib
sys.path.append('/home/dberretta-iit.local/Documents/Repos/hpe-core/')
from pycore.moveenet.visualization import visualization


# Defining a Class
class GraphVisualization:

    def __init__(self, data=None, res=None, pred=None, visualise = None, multiplier=0.6):
        # visual is a list which stores all
        # the set of edges that constitutes a
        # graph
        if res is None:
            self.res = [480, 640]
        else:
            self.res = res
        if data is not None:
            self.x = data.x  # energy of the segment
            self.pos = data.pos.type(torch.int).cpu().numpy()  # location of the node
            self.edge_index = data.edge_index  # Pairs of node indices that make up the edges
            self.edge_attr = data.edge_attr  # length of edges. Not needed for visualisation.
            self.y = np.squeeze(data.y.type(torch.int).cpu().numpy())  # Output
            self.data = data
            if pred is not None:
                self.pred = (pred[0].cpu().numpy()).astype(int)
            else:
                self.pred = None
            self.th_pck = np.squeeze(data.th_pck.cpu().numpy()*multiplier).astype(int)
            try:
                self.contrib = data.contrib
            except:
                self.contrib = None

            self.pos_dict = {}

            for i in range(len(data.pos)):
                self.pos_dict[i] = [data.pos[i, 0], data.pos[i, 1]]

        self.visual = []
        self.visualise = visualise        
    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.visual)
        nx.draw_networkx(G)
        plt.show()

    def create_image(self, show_image=False, show_gt=False, annotation=None, show_pred=False, joints=None, show_avg=False):
        # TODO: Color map for the strength.
        image = np.zeros(self.res, np.uint8)
        thickness = 2
        count = 0
        if joints is not None:
            if joints == 1:
                annotation = 'center'
            elif joints == 13:
                annotation = 'skeleton'
            else:
                print("Joints {} not configured for visualisation".format(joints))
                return 0
        # print('inside create_image, joints is set to: '+str(joints))
        # print('and, annotations is set to: '+annotation)
        for node1, node2 in self.edge_index.T:
            # print(self.x[node1])
            # color = np.int(255 * self.x[node1])
            color = 150
            cv2.line(image, self.pos[node1, :], self.pos[node2, :], (color, color), thickness)
            cv2.circle(image, self.pos[node1, :], radius=3, color=(color, color))
            count += 1
        image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_GRAY2BGR)
        if show_gt:
            if annotation == 'skeleton':
                image = visualization.add_skeleton(image, self.y, (0, 255, 0), lines=True, normalised=False)
                cv2.line(image, (530, 440),(550,440), (0,255,0), thickness) #GT legend
                cv2.putText(image, 'GT', (570, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), int(thickness/2), cv2.LINE_AA)
            elif annotation == 'center':
                cv2.circle(image, (self.y[1], self.y[0]), color=(255, 0, 255), radius=int(self.th_pck))
        if show_pred and self.pred is not None:
            if annotation == 'skeleton':
                image = visualization.add_skeleton(image, self.pred, (255, 0, 0), lines=True, normalised=False)
                #add legend
                cv2.rectangle(image, (520,420), (640,480), (255,255,255), int(thickness/2)) #img size [480, 640]
                cv2.line(image, (530, 460),(550,460), (255,0,0), thickness) #Predicted skeleton
                cv2.putText(image, 'Predict', (570, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), int(thickness/2), cv2.LINE_AA)
            elif annotation == 'center':
                # print(self.th_pck)
                cv2.circle(image, (self.pred[1],self.pred[0]), color=(255, 255, 0), radius=5, thickness=2)
        if show_avg and annotation == 'center':
            avg = np.mean(self.pos, axis=0).astype(np.int)
            # print(avg)
            cv2.circle(image, avg, color=(0, 255, 255), radius=5, thickness=2)

        image = cv2.resize(image, (self.res[1], self.res[0]))
        if show_image:
            cv2.imshow('', image)
            cv2.waitKey(1)
        return image


    def visualise_contribution(self, show_image=False, show_gt=False, annotation=None, show_pred=False, joints=None):
        # TODO: Color map for the strength.
        image = np.zeros(self.res, np.uint8)
        thickness = 2
        # count = 0
        if joints is not None:
            if joints == 1:
                annotation = 'center'
            elif joints == 13:
                annotation = 'skeleton'
            else:
                print("Joints {} not configured for visualisation".format(joints))
                return 0
        # print('inside create_image, joints is set to: '+str(joints))
        # print('and, annotations is set to: '+annotation)
        for node1, node2 in self.edge_index.T:
            # print(self.x[node1])
            color1 = np.int(2550 * torch.sum(self.contrib[node1]))
            color2 = np.int(2550 * torch.sum(self.contrib[node2]))
            cv2.circle(image, self.pos[node1, :], radius=3, color=(color1), thickness=2)
            cv2.circle(image, self.pos[node2, :], radius=3, color=(color2), thickness=2)
            cv2.line(image, self.pos[node1, :], self.pos[node2, :], (np.int(255 * self.x[node1])), thickness)
            # count += 1
        image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_GRAY2BGR)
        if show_gt:
            if annotation == 'skeleton':
                image = visualization.add_skeleton(image, self.y, (255, 0, 0), lines=True, normalised=False)
            elif annotation == 'center':
                cv2.circle(image, self.y, color=(256, 0, 255), radius=int(self.th_pck))
        if show_pred and self.pred is not None:
            if annotation == 'skeleton':
                image = visualization.add_skeleton(image, self.pred, (255, 0, 0), lines=True, normalised=False)
            elif annotation == 'center':
                # print(self.th_pck)
                cv2.circle(image, self.pred, color=(256, 256, 0), radius=3)

        image = cv2.resize(image, (self.res[1], self.res[0]))
        if show_image:
            cv2.imshow('', image)
            cv2.waitKey(1)
        return image


    def visualise_vectors(self, show_image=False, show_gt=False, annotation=None, show_pred=False, joints=None, histogram = False,top_nodes = False, node_loss = False):
        # TODO: Color map for the strength.
        image = np.zeros(self.res, np.uint8)
        thickness = 2
        count = 0
        if self.visualise == 'vectors-head':
            j = visualization.hpecore_kps_labels.get('head') # {'head':0, 'handR': 7, 'handL': 8}
        elif self.visualise == 'vectors-handR':
            j = visualization.hpecore_kps_labels.get('handR')

        top_contrib_nodes = int(0.3* len(self.pos)) # top contrib nodes
        if joints is not None:
            if joints == 1:
                annotation = 'center'
            elif joints == 13:
                annotation = 'skeleton'
            else:
                print("Joints {} not configured for visualisation".format(joints))
                return 0
        for node1, node2 in self.edge_index.T:
            # print(self.x[node1])
            # color = np.int(255 * self.x[node1])
            color = 150
            cv2.line(image, self.pos[node1, :], self.pos[node2, :], (color, color), thickness)
            cv2.circle(image, self.pos[node1, :], radius=3, color=(color, color))

        image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_GRAY2BGR)

        out_xs = torch.zeros(2).type_as(self.contrib)

        if node_loss:
            if top_nodes:
                top_values_y, indices_y = torch.topk(torch.abs(self.contrib[:,4*j+2:4*j+3].squeeze()), k = top_contrib_nodes, largest=True, sorted=True)
                top_values_x, indices_x = torch.topk(torch.abs(self.contrib[:,4*j+3:4*j+4].squeeze()), k = top_contrib_nodes, largest=True, sorted=True)   
            
            # point of the weighted sum of the top 30% contrib nodes - I am sure this is a correct calculation
                out_xs[0] = torch.mean(self.data.pos[indices_x, 1]+ self.contrib[indices_x, 4*j+1]*self.res[0])
                out_xs[1] = torch.mean(self.data.pos[indices_y, 0]+ self.contrib[indices_y, 4*j]*self.res[1])
                
            else:
            # print(out_xs) y, x
                out_xs[0] = torch.mean(self.data.pos[:, 1] + self.contrib[:, 4 * j + 1]*self.res[0])
                out_xs[1] = torch.mean(self.data.pos[:, 0] + self.contrib[:, 4 * j]*self.res[1])
        
        else:
            if top_nodes:
                top_values_y, indices_y = torch.topk(torch.abs(self.contrib[:,4*j+2:4*j+3].squeeze()), k = top_contrib_nodes, largest=True, sorted=True)
                top_values_x, indices_x = torch.topk(torch.abs(self.contrib[:,4*j+3:4*j+4].squeeze()), k = top_contrib_nodes, largest=True, sorted=True)   
            
            # point of the weighted sum of the top 30% contrib nodes - I am sure this is a correct calculation
                out_xs[0] = torch.dot(self.contrib[indices_x, 4*j+3], self.data.pos[indices_x, 1]+ self.contrib[indices_x, 4*j+1]*self.res[0])
                out_xs[1] = torch.dot(self.contrib[indices_y, 4*j+2], self.data.pos[indices_y, 0]+ self.contrib[indices_y, 4*j]*self.res[1])
                
            else:
            # print(out_xs) y, x
                out_xs[0] = torch.dot(self.contrib[:, 4 * j + 3], self.data.pos[:, 1] + self.contrib[:, 4 * j + 1]*self.res[0])
                out_xs[1] = torch.dot(self.contrib[:, 4 * j + 2], self.data.pos[:, 0] + self.contrib[:, 4 * j]*self.res[1])
        
        points_contrib_all_node = out_xs.detach().cpu().numpy().astype(int)
        # print('point of weighted sum of top 10: ', points_contrib_all_node)
        # exit()
        # Histogram of the weights
        weight_xs = self.contrib[:,4*j+3].detach().cpu().numpy()
        weight_ys = self.contrib[:,4*j+2].detach().cpu().numpy()

        weight_xs_normalized = cv2.normalize(weight_xs, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        weight_ys_normalized = cv2.normalize(weight_ys, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if histogram == True:
            fig, axs = plt.subplots(1,2, layout='constrained',sharey = True)
            axs[0].hist(weight_xs, bins = 50)
            axs[0].set_title('weighted x')
            axs[1].hist(weight_ys, bins = 50)
            axs[1].set_title('weighted y')
            fig.suptitle('Histogram of weights contribute to the hand joint', fontsize = 16)
            # plt.hist(weight_ys, bins = 100)
            plt.show()
            # print('sum weighted x: ', np.sum(weight_xs))
            # print('sum weighted y: ', np.sum(weight_ys))
        # exit()
        for i, cx_norm in enumerate(weight_ys_normalized):

            gray_x = np.array([cx_norm*255], dtype=np.uint8)
            # print('color gray:', gray_x)
            color_mapped = cv2.applyColorMap(gray_x,cv2.COLORMAP_PINK)
            # print('color map:', color_mapped)
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = tuple(int(c) for c in color_mapped[0, 0])
            vector = (np.multiply(self.contrib[i, 4*j:4*j+2].detach().cpu().numpy(), self.res[1::-1])).astype(int)  
            # # Get the top 10 values and their indices
            if top_nodes:
                    #Plot top k% of contrib node
                if torch.isin(torch.tensor(i, device='cuda:0'), indices_x[0:2]) or torch.isin(torch.tensor(i, device='cuda:0'), indices_y[0:]):
                    cv2.line(image, self.pos[i, :], [self.pos[i, 1] + vector[1],self.pos[i, 0] + vector[0]], color, int(thickness))
                    image = cv2.putText(image, str(vector), self.pos[i, :], font, 0.5, (0,255,0), thickness, cv2.LINE_AA)
                        
            else:
            #Plot all nodes contributing to the final joints
                cv2.line(image, self.pos[i, :], [self.pos[i, 1] + vector[1],self.pos[i, 0] + vector[0]], color, int(thickness/2))
            count +=1
                
        cv2.circle(image, points_contrib_all_node, radius=3, color=(255,255,0), thickness=5)
        
        if show_gt:
            if annotation == 'skeleton':
                image = visualization.add_skeleton(image, self.y, (0, 255, 0), lines=True, normalised=False)
                cv2.line(image, (530, 440),(550,440), (0,255,0), thickness) #GT legend
                cv2.putText(image, 'GT', (570, 440), font, 0.5, (0,255,0), int(thickness/2), cv2.LINE_AA)
            elif annotation == 'center':
                cv2.circle(image, self.y, color=(256, 0, 255), radius=int(self.th_pck))
        if show_pred and self.pred is not None:
            if annotation == 'skeleton':
                image = visualization.add_skeleton(image, self.pred, (255, 0, 0), lines=True, normalised=False)
                #add legend
                cv2.rectangle(image, (520,420), (640,480), (255,255,255), int(thickness/2)) #img size [480, 640]
                cv2.line(image, (530, 460),(550,460), (255,0,0), thickness) #Predicted skeleton
                cv2.putText(image, 'Predict', (570, 460), font, 0.5, (255,0,0), int(thickness/2), cv2.LINE_AA)
            elif annotation == 'center':
                # print(self.th_pck)
                cv2.circle(image, self.pred, color=(256, 256, 0), radius=3)

        image = cv2.resize(image, (self.res[1], self.res[0]))
        if show_image:
            cv2.imshow('', image)
            cv2.waitKey(1)
        return image

def visualise_single_selective_pooling_weight_vectors(self, show_image=False, show_gt=False, annotation=None, show_pred=False, joints=None, histogram = False,top_nodes = False, node_loss = False):
        # TODO: Color map for the strength.
        image = np.zeros(self.res, np.uint8)
        thickness = 2
        count = 0
        if self.visualise == 'vectors-head':
            j = visualization.hpecore_kps_labels.get('head') # {'head':0, 'handR': 7, 'handL': 8}
        elif self.visualise == 'vectors-handR':
            j = visualization.hpecore_kps_labels.get('handR')

        top_contrib_nodes = int(0.3* len(self.pos)) # top contrib nodes
        if joints is not None:
            if joints == 1:
                annotation = 'center'
            elif joints == 13:
                annotation = 'skeleton'
            else:
                print("Joints {} not configured for visualisation".format(joints))
                return 0
        for node1, node2 in self.edge_index.T:
            # print(self.x[node1])
            # color = np.int(255 * self.x[node1])
            color = 150
            cv2.line(image, self.pos[node1, :], self.pos[node2, :], (color, color), thickness)
            cv2.circle(image, self.pos[node1, :], radius=3, color=(color, color))

        image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_GRAY2BGR)

        out_xs = torch.zeros(2).type_as(self.contrib)

        if node_loss:
            if top_nodes:
                # top_values_y, indices_y = torch.topk(torch.abs(self.contrib[:,4*j+2:4*j+3].squeeze()), k = top_contrib_nodes, largest=True, sorted=True)
                # top_values_x, indices_x = torch.topk(torch.abs(self.contrib[:,4*j+3:4*j+4].squeeze()), k = top_contrib_nodes, largest=True, sorted=True)
                top_values_y, indices_y = torch.topk(torch.abs(self.contrib[:,4*j+2:4*j+3].squeeze()), k = top_contrib_nodes, largest=True, sorted=True)
                top_values_x, indices_x = torch.topk(torch.abs(self.contrib[:,4*j+2:4*j+3].squeeze()), k = top_contrib_nodes, largest=True, sorted=True)    
            
            # point of the weighted sum of the top 30% contrib nodes - I am sure this is a correct calculation
                out_xs[0] = torch.mean(self.data.pos[indices_x, 1]+ self.contrib[indices_x, 4*j+1]*self.res[0])
                out_xs[1] = torch.mean(self.data.pos[indices_y, 0]+ self.contrib[indices_y, 4*j]*self.res[1])
                
            else:
            # print(out_xs) y, x
                out_xs[0] = torch.mean(self.data.pos[:, 1] + self.contrib[:, 4 * j + 1]*self.res[0])
                out_xs[1] = torch.mean(self.data.pos[:, 0] + self.contrib[:, 4 * j]*self.res[1])
        
        else:
            if top_nodes:
                # top_values_y, indices_y = torch.topk(torch.abs(self.contrib[:,4*j+2:4*j+3].squeeze()), k = top_contrib_nodes, largest=True, sorted=True)
                # top_values_x, indices_x = torch.topk(torch.abs(self.contrib[:,4*j+3:4*j+4].squeeze()), k = top_contrib_nodes, largest=True, sorted=True)
                top_values, indices = torch.topk(torch.abs(self.contrib[:,4*j+2:4*j+3].squeeze()), k = top_contrib_nodes, largest=True, sorted=True)
            
            # point of the weighted sum of the top 30% contrib nodes - I am sure this is a correct calculation
                # out_xs[0] = torch.dot(self.contrib[indices_y, 4*j+3], self.data.pos[indices_y, 1]+ self.contrib[indices_y, 4*j+1]*self.res[0])
                # out_xs[1] = torch.dot(self.contrib[indices_x, 4*j+2], self.data.pos[indices_x, 0]+ self.contrib[indices_x, 4*j]*self.res[1])
                out_xs[0] = torch.dot(self.contrib[indices, 4*j+2], self.data.pos[indices, 1]+ self.contrib[indices, 4*j+1]*self.res[0])
                out_xs[1] = torch.dot(self.contrib[indices, 4*j+2], self.data.pos[indices, 0]+ self.contrib[indices, 4*j]*self.res[1])
                
            # print(out_xs) y, x
            out_xs[0] = torch.dot(self.contrib[:, 4 * j + 2], self.data.pos[:, 1] + self.contrib[:, 4 * j + 1]*self.res[0])
            out_xs[1] = torch.dot(self.contrib[:, 4 * j + 2], self.data.pos[:, 0] + self.contrib[:, 4 * j + 0]*self.res[1])
        
        points_contrib_all_node = out_xs.detach().cpu().numpy().astype(int)
        # print('point of weighted sum of top 10: ', points_contrib_all_node)
        # exit()
        # Histogram of the weights
        weight = self.contrib[:,4*j+2].detach().cpu().numpy()
        weight_ys = self.contrib[:,4*j+3].detach().cpu().numpy()
        # print('weighted x: ', weight_xs)
        # print('weighted y: ', weight_ys)
        # weight_xs_normalized = cv2.normalize(weight_xs, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # weight_ys_normalized = cv2.normalize(weight_ys, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        weight_normalized = cv2.normalize(weight, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        if histogram == True:
            fig, axs = plt.subplots(1,2, layout='constrained',sharey = True)
            axs[0].hist(weight, range = (1e-9, weight.max()), bins = 200)
            axs[0].set_title('weighted x')
            axs[1].hist(weight_ys,range = (1e-9, weight.max()),  bins = 200)
            axs[1].set_title('weighted y')
            fig.suptitle('Histogram of weights contribute to the hand joint', fontsize = 16)
            # plt.hist(weight_ys, bins = 100)
            plt.show()
        for i, cx_norm in enumerate(weight_normalized):
            gray_x = np.array([cx_norm*255], dtype=np.uint8)
            # print('color gray:', gray_x)
            color_mapped = cv2.applyColorMap(gray_x,cv2.COLORMAP_PINK)
            # print('color map:', color_mapped)
            font = cv2.FONT_HERSHEY_SIMPLEX
            weight_color = tuple(int(c) for c in color_mapped[0, 0])
            vector = (np.multiply(self.contrib[i, 4*j:4*j+2].detach().cpu().numpy(), self.res[1::-1])).astype(int) 
            # # Get the top 10 values and their indices
            if top_nodes:
                    #Plot top k% of contrib node
                if torch.isin(torch.tensor(i, device='cuda:0'), indices[0:2]):
                    cv2.line(image, self.pos[i, :], [self.pos[i, 1] + vector[1],self.pos[i, 0] + vector[0]], weight_color, int(thickness/2))
                    # image = cv2.putText(image, str(vector), self.pos[i, :], font, 0.5, (0,255,0), thickness, cv2.LINE_AA)
                        
            else:
            #Plot all nodes contributing to the final joints
                cv2.line(image, self.pos[i, :], [self.pos[i, 1] + vector[1],self.pos[i, 0] + vector[0]], weight_color, int(thickness/2))
            count +=1
                
        cv2.circle(image, points_contrib_all_node, radius=3, color=(255,255,0), thickness=5)
        
        if show_gt:
            if annotation == 'skeleton':
                image = visualization.add_skeleton(image, self.y, (0, 255, 0), lines=True, normalised=False, flip=False)
                cv2.line(image, (530, 440),(550,440), (0,255,0), thickness) #GT legend
                cv2.putText(image, 'GT', (570, 440), font, 0.5, (0,255,0), int(thickness/2), cv2.LINE_AA)
            elif annotation == 'center':
                cv2.circle(image, self.y, color=(256, 0, 255), radius=int(self.th_pck))
        if show_pred and self.pred is not None:
            if annotation == 'skeleton':
                image = visualization.add_skeleton(image, self.pred, (255, 0, 0), lines=True, normalised=False, flip=False)
                #add legend
                cv2.rectangle(image, (520,420), (640,480), (255,255,255), int(thickness/2)) #img size [480, 640]
                cv2.line(image, (530, 460),(550,460), (255,0,0), thickness) #Predicted skeleton
                cv2.putText(image, 'Predict', (570, 460), font, 0.5, (255,0,0), int(thickness/2), cv2.LINE_AA)
            elif annotation == 'center':
                # print(self.th_pck)
                cv2.circle(image, self.pred, color=(256, 256, 0), radius=3)

        image = cv2.resize(image, (self.res[1], self.res[0]))
        if show_image:
            cv2.imshow('', image)
            cv2.waitKey(1)
        return image

def visualize_losses(losses):
    x = [i for i in range(losses.shape[1])]
    plt.plot(x[1:], losses[0, 1:], label='Training loss')
    plt.plot(x[1:], losses[1, 1:], label='Validation loss')
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss with an simple GNN on DHP19 for head detection");
    plt.show()


class CloseEvent(object):

    def __init__(self):
        self.first = True

    def __call__(self):
        if self.first:
            self.first = False
            return
        plt.close()


if __name__ == '__main__':
    # G = nx.complete_graph(5)
    # nx.draw(G)
    # Driver code
    G = GraphVisualization()
    # G.add_all_edges()

    G.visualize()

