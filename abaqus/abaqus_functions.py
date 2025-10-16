from abaqus import *
import testUtils
testUtils.setBackwardCompatibility()
from abaqusConstants import *
import __main__
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
import numpy as np
import random
random.seed(1111)

session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)

def amplitude(model, name_part, NameRef1, NameRef2, amp, num_steps, exp_fact, num_csv_TP, num_odb_TP, step_name='step-', start_point=0, end_point=1, amp_tuple=0, maxU=25, offset=0, radius=None):
    # Exp_fact is numbers from 1 (linear ramp) to 100+ (exponential)
    #step_name = step_name+name_part

    if amp_tuple != 0:
        model.TabularAmplitude(data=amp_tuple, name=
            amp, timeSpan=STEP)
    else:
        amp_data = exp_amp(num_steps, exp_fact, start_point, end_point, step_name, offset)
        model.TabularAmplitude(data=amp_data, name=
            amp, timeSpan=STEP)

    #Increase minimum number of attempts from 5 to 30
    if maxU == maxU:
        model.steps[step_name].control.setValues(allowPropagation=OFF, 
            resetDefaultValues=OFF, timeIncrementation=(4.0, 8.0, 9.0, 16.0, 10.0, 4.0, 
            12.0, 25, 6.0, 3.0, 50.0))

    model.fieldOutputRequests['F-Output-1'].setValues(timePoint='TimePoints-ODB')

    model.HistoryOutputRequest(createStepName=step_name, name=
        'H-Output-2-'+step_name, rebar=EXCLUDE, region=   
        model.rootAssembly.sets[NameRef1], sectionPoints=DEFAULT, 
        variables=('U1','U2','U3','RF1','RF2','RF3','CF1','CF2','CF3'),timePoint='TimePoints-CSV')

    if radius is None:
        model.HistoryOutputRequest(createStepName=step_name, name=
            'H-Output-3-'+step_name, rebar=EXCLUDE, region=   
            model.rootAssembly.sets[NameRef2], sectionPoints=DEFAULT, 
            variables=('U1','U2','U3','RF1','RF2','RF3','CF1','CF2','CF3'),timePoint='TimePoints-CSV')

    model.fieldOutputRequests['F-Output-1'].setValues(variables=(
    'UR','S', 'PE', 'PEEQ', 'PEMAG', 'LE', 'U', 'V', 'A', 'RF', 'CF', 'CSTRESS', 
    'CDISP', 'STH', 'SE','COORD','EVOL'),timePoint='TimePoints-ODB')

    amp_csv_TP = np.linspace(0,1,num_csv_TP)
    amp_odb_TP = np.linspace(0,1,num_odb_TP)

    time_csv_TP = num_csv_TP*[0]
    time_odb_TP = num_odb_TP*[0]
    if exp_fact == 1:
        for i,t in enumerate(np.linspace(0,1,num_csv_TP)):
            time_csv_TP[i] = ((float(t)),)
        for i,t in enumerate(np.linspace(0,1,num_odb_TP)):
            time_odb_TP[i] = ((float(t)),) 
    else:
        for i in range(num_csv_TP):
                if i != 0:
                    time_csv_TP[i] = (((np.log10(amp_csv_TP[i]*((exp_fact)-1)+1))/np.log10(exp_fact)),)
                else:
                    time_csv_TP[i] = ((0),)
        for i in range(num_odb_TP):
                if i != 0:
                    time_odb_TP[i] = (((np.log10(amp_odb_TP[i]*((exp_fact)-1)+1))/np.log10(exp_fact)),)
                else:
                    time_odb_TP[i] = ((0),)
    model.TimePoint(name='TimePoints-CSV', points=tuple(time_csv_TP))
    model.TimePoint(name='TimePoints-ODB', points=tuple(time_odb_TP))

def exp_amp(num_timesteps, b, start_point, end_point, step_name, offset):
    def exp_fxn(x, b):
        if b == 1:
            return x
        else:
            return 1./(b-1) * (b**x - 1)

    all_t = np.linspace(start_point, end_point, num_timesteps)
    all_tuples = (num_timesteps) * [0]
    for i in range(num_timesteps):
        all_tuples[i] = (all_t[i], exp_fxn(all_t[i], b) + offset)
    return all_tuples

def generate_RP(model, dimension,NameRef1, NameRef2):
    model.Part(dimensionality=dimension, name=NameRef1, type=DEFORMABLE_BODY)
    model.parts[NameRef1].ReferencePoint(point=(0.0, 0.0, 0.0))
    model.Part(dimensionality=dimension, name=NameRef2, type=DEFORMABLE_BODY)
    model.parts[NameRef2].ReferencePoint(point=(0.0, 0.0, 0.0))
    model.rootAssembly.Instance(dependent=ON, name=NameRef1, part=model.parts[NameRef1])
    model.rootAssembly.Instance(dependent=ON, name=NameRef2, part=model.parts[NameRef2])
    # Create set of reference points
    model.rootAssembly.Set(name=NameRef1, referencePoints=(
        model.rootAssembly.instances[NameRef1].referencePoints[1],))
    model.rootAssembly.Set(name=NameRef2, referencePoints=(
        model.rootAssembly.instances[NameRef2].referencePoints[1],))


def NodeSequence(model, name_part, node):
    # Example of Abaqus being dumb:
    # When creating a set, abaqus wants node objects, however
    # if given a node object as instancePart[2] it will fail
    # creation. Otherwise instancePart[2:3] is the correct version
    instancePart = model.rootAssembly.instances[name_part+'-1'].nodes
    return instancePart[node.label-1:node.label]

def PBC_planar(model, dimension, name_part, latticeVec1, latticeVec2, NameRef1, NameRef, NameCornerNodes, NameEdgeNodes):
    #Variables
    # latticeVec1 & latticeVec2 two dimensional list of x and y coordinates
    # exp: latticeVec1 = [x0,y0]
    # NameRef1 and NameRef, names of the reference points to which constraints will be tied
    # NameCornerNodes, name of the set for corner nodes
    # NameEdgeNodes, name of the set for all edge nodes (including corner)
    ############################
    # Pre-Processing
    ############################
    # Note: lattice-vectors must be provided in standard PBC
    LatticeVec = (latticeVec1,latticeVec2)
    # Nodes
    corner_nodes = model.rootAssembly.sets[name_part+'-1.'+NameCornerNodes].nodes
    edge_nodes = model.rootAssembly.sets[name_part+'-1.'+NameEdgeNodes].nodes
    # Create a set for the origin nodes
    print('Corner Nodes:')
    for node in corner_nodes:
        print(node.coordinates)
        if node.coordinates == (0,0,0):
            model.rootAssembly.Set(name='corner-origin', nodes=NodeSequence(model, name_part, node))
            origin_node = model.rootAssembly.sets['corner-origin'].nodes
    if dimension == THREE_D:
        dim_matrix = [3, 4, 5, 6]
    elif dimension == TWO_D_PLANAR:
        dim_matrix = []
    else:
        raise BaseException('Error: Dimension not defined for PBC.')

    ############################
    # Corner Boundary Conditions
    ############################
    usedNodes = []
    repConst = 1
    usedNodes.append(origin_node[0])
    # For every other corner node, tie it back to the origin.
    for node in corner_nodes:
        if node not in origin_node:
            coordX = node.coordinates[0]    # X-coordinate: Always length dx with respect to origin
            coordY = node.coordinates[1]    # Y-coordinate: Always length dy wtih respect to origin
            # Create a set with current node to be used in equations
            model.rootAssembly.Set(name='corner-node-' + str(
                repConst), nodes=NodeSequence(model, name_part, node))
            for Dim1 in [1,2]:          # Create planar equations
                model.Equation(name='corner-plrDOF-' + str(repConst) + '-' + str(Dim1),
                                            terms=((-1.0, 'corner-node-' + str(repConst), Dim1), (1.0, 'corner-origin', Dim1),
                                                    (coordX, NameRef1, Dim1), (coordY, NameRef, Dim1)))
            for Dim1 in dim_matrix:     # Create equality equations
                model.Equation(name='corner-rotDOF-' + str(Dim1) + '-' + str(repConst),
                                            terms=((-1.0, 'corner-node-' + str(repConst), Dim1), (1.0, 'corner-origin', Dim1)))    
            repConst = repConst + 1     # Increase integer for naming equation constraint
            usedNodes.append(node)

    ############################
    # Edge Boundary Conditions
    ############################
    repConst = 1
    for node1 in edge_nodes:
        if node1 not in corner_nodes:
            if node1 in usedNodes:
                pass
            else:
                stop = False
                # Find Node1 Coordinates
                coordX1 = node1.coordinates[0]
                coordY1 = node1.coordinates[1]
                for node2 in edge_nodes:
                    if node2 not in corner_nodes:
                        # Find Node2 Coordinates
                        coordX2 = node2.coordinates[0]
                        coordY2 = node2.coordinates[1]
                        # Find distance between nodes
                        dx = coordX2 - coordX1  # X-Distance between nodes
                        dy = coordY2 - coordY1  # Y-Distance between nodes
                        for vec in LatticeVec:
                            tol = 1e-2
                            if abs(vec[0] - dx) < tol and abs(vec[1] - dy) < tol:
                                if stop:
                                    break
                                else:
                                    # Correct combination found begin creating sets for use in equations constraints
                                    model.rootAssembly.Set(name='Node-1-' + str(
                                        repConst), nodes=NodeSequence(model, name_part, node1))
                                    model.rootAssembly.Set(name='Node-2-' + str(
                                        repConst), nodes=NodeSequence(model, name_part, node2))
                                    for Dim1 in [1, 2]:
                                        model.Equation(name='node-plrDOF-' + str(repConst) + '-' + str(Dim1),
                                                                    terms=((1.0, 'Node-1-' + str(repConst), Dim1), (-1.0, 'Node-2-' + str(repConst), Dim1),
                                                                            (dx, NameRef1, Dim1), (dy, NameRef, Dim1)))
                                    for Dim1 in dim_matrix:
                                        model.Equation(name='node-rotDOF-' + str(repConst) + '-' + str(Dim1),
                                                                    terms=((-1.0, 'Node-1-' + str(repConst), Dim1), (1.0, 'Node-2-' + str(repConst), Dim1)))   
                                                                        # Remove used node from available list
                                    usedNodes.append(node1)
                                    usedNodes.append(node2)
                                    repConst = repConst + 1  # Increase integer for naming equation constraint
                                    # Don't look further, go to following node.
                                    stop = True
    
    if PBC_check(usedNodes, edge_nodes):
        raise BaseException("Error: PBC not implemeneted correctly, stopping python script.")

    print('Periodic Boundary Conditions successfully implemented')
    return 

def PBC_check(used_nodes, edge_nodes):
    node_check = np.zeros(len(edge_nodes))

    for i,node in enumerate(edge_nodes):
        if node in used_nodes:
            node_check[i] = 1
        else:
            print('Node PBC: ')
            print(node)

    if 0 in node_check:
        return True
    else:
        return False