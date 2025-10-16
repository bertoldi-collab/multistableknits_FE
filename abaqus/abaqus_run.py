################################################
# Import Library
################################################

from abaqus import mdb, session
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
from numpy import linalg as LA
import math
import time
import itertools
from itertools import compress
import os
import random

class BuckleSimulation:
    def __init__(self, a, b, d, thickness, thickness_ratio, xreps, yreps, exp_coeff11, exp_coeff22, nu, E, mesh_size=None, pattern=None, direction=None, deformation=None, simulation_time=2.0):
        self.a = a
        self.b = b
        self.d = d
        self.thickness = thickness
        self.thickness_ratio = thickness_ratio
        self.xreps = xreps
        self.yreps = yreps
        self.exp_coeff11 = exp_coeff11
        self.exp_coeff22 = exp_coeff22
        self.nu = nu
        self.E = E
        self.mesh_size = mesh_size if mesh_size else np.min([a, b, d])/7
        self.pattern = pattern
        self.direction = direction
        self.deformation = deformation
        self.simulation_time = simulation_time
        self.model = 'Model-1'
        self.sketch = 'Sketch-1'
        self.material_base = 'Material-1'
        self.section = 'Section-1'
        self.job = 'Job-1'
        self.unitwidth = 2*a
        self.unitheight = (d+b)*2
        self.tol = 0.01
        self.angle_part = 0
        self.theta_part = self.angle_part * np.pi/180.0
        self.original_dir = os.getcwd()
        self.buckle_step = False

    def setup_directory(self):
        job_folder = "job_{}_point_final_long_{}".format(self.pattern, self.E)
        print("Creating job folder: {}".format(job_folder))
        try:
            if not os.path.exists(job_folder):
                os.makedirs(job_folder)
            self.original_dir = os.getcwd()
            os.chdir(job_folder)
            print("Working in directory: {}".format(os.getcwd()))
        except OSError as e:
            print("Error creating or accessing job folder: {}".format(e))
            raise

    def create_part(self):
        self.unitwidth = 2*self.a
        self.unitheight = (self.d+self.b)*2
        session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)
        self.m = mdb.Model(name=self.model)
        self.m.rootAssembly.DatumCsysByDefault(CARTESIAN)
        s = self.m.ConstrainedSketch(name='base_part', sheetSize=200.0)
        g, v, dp, c = s.geometry, s.vertices, s.dimensions, s.constraints
        bot_line = self.m.sketches['base_part'].Line(point1=(0.0, 0.0), point2=(self.unitwidth*self.xreps, 0.0))
        right_line = self.m.sketches['base_part'].Line(point1=(self.unitwidth*self.xreps, 0.0), point2=(self.unitwidth*self.xreps, self.unitheight*self.yreps))
        top_line = self.m.sketches['base_part'].Line(point1=(self.unitwidth*self.xreps, self.unitheight*self.yreps), point2=(0.0, self.unitheight*self.yreps))
        left_line = self.m.sketches['base_part'].Line(point1=(0.0, self.unitheight*self.yreps), point2=(0.0, 0.0))
        botPoint = bot_line.pointOn+(0,)
        rightPoint = right_line.pointOn+(0,)
        topPoint = top_line.pointOn+(0,)
        leftPoint = left_line.pointOn+(0,)
        self.Part_1 = self.m.Part(dimensionality=THREE_D, name='Part-1', type=DEFORMABLE_BODY)
        self.m.parts['Part-1'].BaseShell(sketch=self.m.sketches['base_part'])
        p = self.m.parts['Part-1']
        e = p.edges
        p.Set(edges=e,  name='edgesAll')
        p.Set(edges=p.edges.findAt((botPoint, )), name='edge_bot')
        p.Set(edges=p.edges.findAt((rightPoint, )), name='edge_right')
        p.Set(edges=p.edges.findAt((topPoint, )), name='edge_top')
        p.Set(edges=p.edges.findAt((leftPoint, )), name='edge_left')
        mesh_tol = self.mesh_size/4
        xrec = [0, self.unitwidth*self.xreps, self.unitwidth*self.xreps, 0]
        yrec = [0, 0, self.unitheight*self.yreps, self.unitheight*self.yreps]
        corner_TL = p.Set(name='cn_TL', vertices=p.vertices.getByBoundingBox(xrec[3]-mesh_tol, yrec[3]-mesh_tol, -mesh_tol, xrec[3]+mesh_tol, yrec[3]+mesh_tol, mesh_tol))
        corner_TR = p.Set(name='cn_TR', vertices=p.vertices.getByBoundingBox(xrec[2]-mesh_tol, yrec[2]-mesh_tol, -mesh_tol, xrec[2]+mesh_tol, yrec[2]+mesh_tol, mesh_tol))
        corner_BR = p.Set(name='cn_BR', vertices=p.vertices.getByBoundingBox(xrec[1]-mesh_tol, yrec[1]-mesh_tol, -mesh_tol, xrec[1]+mesh_tol, yrec[1]+mesh_tol, mesh_tol))
        corner_BL = p.Set(name='origin', vertices=p.vertices.getByBoundingBox(xrec[0]-mesh_tol, yrec[0]-mesh_tol, -mesh_tol, xrec[0]+mesh_tol, yrec[0]+mesh_tol, mesh_tol))
        self.m.ConstrainedSketch(name='partition_sketch', sheetSize=200.0)
        set1_centers = []
        set2_centers = []
        for x_offset in range(0,self.xreps):
            for y_offset in range(0, self.yreps):
                unit_x = x_offset * self.unitwidth
                unit_y = y_offset * self.unitheight
                s = self.m.sketches['partition_sketch']
                if self.pattern=='H-shaped':
                    for i in range(2):
                        for j in range(4):
                            x_start = i * self.a + unit_x
                            if j== 0:
                                y_start = 0 + unit_y
                                height = self.d
                            elif j == 1:
                                y_start = self.d + unit_y
                                height = self.b
                            elif j == 2:
                                y_start = self.b + self.d + unit_y
                                height = self.d
                            elif j == 3:
                                y_start = self.b + self.d*2 + unit_y
                                height = self.b
                            s.rectangle(point1=(x_start, y_start), point2=(x_start + self.a, y_start + height))
                            center_x = x_start + self.a/2
                            center_y = y_start + height/2
                            if j == 3:
                                set1_centers.append((center_x, center_y, 0))
                            elif i == 1:
                                set2_centers.append((center_x, center_y, 0))
                            elif j == 1:
                                set2_centers.append((center_x, center_y, 0))
                            else:
                                set1_centers.append((center_x, center_y, 0))
                elif self.pattern=='Step':
                    if self.xreps % 2 != 0:
                        raise ValueError("xreps must be divisible by 2 for Step pattern")
                    else:
                        for i in range(2):
                            for j in range(4):
                                x_start = i * self.a + unit_x
                                if j== 0:
                                    y_start = 0 + unit_y
                                    height = self.d
                                elif j == 1:
                                    y_start = self.d + unit_y
                                    height = self.b
                                elif j == 2:
                                    y_start = self.b + self.d + unit_y
                                    height = self.d
                                elif j == 3:
                                    y_start = self.b + self.d*2 + unit_y
                                    height = self.b
                                s.rectangle(point1=(x_start, y_start), point2=(x_start + self.a, y_start + height))
                                center_x = x_start + self.a/2
                                center_y = y_start + height/2
                                if (x_offset+1) % 2 ==0:
                                    if j == 0:
                                        set2_centers.append((center_x, center_y, 0))
                                    elif j == 1 and i == 0:
                                        set2_centers.append((center_x, center_y, 0))
                                    elif j == 3 and i == 1:
                                        set2_centers.append((center_x, center_y, 0))
                                    else:
                                        set1_centers.append((center_x, center_y, 0))
                                else:
                                    if j == 0:
                                        set2_centers.append((center_x, center_y, 0))
                                    elif j == 1 and i == 1:
                                        set2_centers.append((center_x, center_y, 0))
                                    elif j == 3 and i == 0:
                                        set2_centers.append((center_x, center_y, 0))
                                    else:
                                        set1_centers.append((center_x, center_y, 0))
                elif self.pattern=='Windmill':
                    if self.xreps % 2 != 0:
                        raise ValueError("xreps must be divisible by 2 for Step pattern")
                    else:
                        for i in range(2):
                            for j in range(4):
                                x_start = i * self.a + unit_x
                                if j== 0:
                                    y_start = 0 + unit_y
                                    height = self.d
                                elif j == 1:
                                    y_start = self.d + unit_y
                                    height = self.b
                                elif j == 2:
                                    y_start = self.b + self.d + unit_y
                                    height = self.d
                                elif j == 3:
                                    y_start = self.b + self.d*2 + unit_y
                                    height = self.b
                                s.rectangle(point1=(x_start, y_start), point2=(x_start + self.a, y_start + height))
                                center_x = x_start + self.a/2
                                center_y = y_start + height/2
                                if (x_offset+1) % 2 ==0:
                                    if j == 0:
                                        set2_centers.append((center_x, center_y, 0))
                                    elif j == 2 and i == 0:
                                        set2_centers.append((center_x, center_y, 0))
                                    elif j == 3 and i == 0:
                                        set2_centers.append((center_x, center_y, 0))
                                    else:
                                        set1_centers.append((center_x, center_y, 0))
                                else:
                                    if j == 0 and i == 1:
                                        set2_centers.append((center_x, center_y, 0))
                                    elif j == 1 and i == 1:
                                        set2_centers.append((center_x, center_y, 0))
                                    elif j == 3:
                                        set2_centers.append((center_x, center_y, 0))
                                    else:
                                        set1_centers.append((center_x, center_y, 0))

        self.Part_1.PartitionFaceBySketch(faces= self.Part_1.faces.findAt(((0.0, 0.0, 0.0),)), sketch=self.m.sketches['partition_sketch'])
        set1 = []
        set2 = []
        for center in set1_centers:
            face = self.Part_1.faces.findAt((center, ))
            if face:
                set1.append(face)
        for center in set2_centers:
            face = self.Part_1.faces.findAt((center, ))
            if face:
                set2.append(face)
        self.Part_1.Set(name = 'Set-1',faces = tuple(set1))
        self.Part_1.Set(name = 'Set-2',faces = tuple(set2))
        self.Part_1.Set(name = 'All',faces = self.Part_1.faces)

    def assign_material(self,exp_coeff11=None, exp_coeff22=None,alpha=4.0):
        self.m.Material(name='Material-1')
        self.m.materials['Material-1'].Elastic(table=((self.E, self.nu), ))
        if exp_coeff22 is not None:
            self.m.materials['Material-1'].Expansion(table=((0.0, exp_coeff22, 0.0), ), type=ORTHOTROPIC)
        else:
            self.m.materials['Material-1'].Expansion(table=((0.0, self.exp_coeff22, 0.0), ), type=ORTHOTROPIC)
        self.m.materials['Material-1'].Density(table=((1e-9, ), ))
        self.m.materials['Material-1'].Damping(alpha=alpha)
        self.m.Material(name='Material-2')
        self.m.materials['Material-2'].Elastic(table=((self.E, self.nu), ))
        if exp_coeff11 is not None:
            self.m.materials['Material-2'].Expansion(table=((exp_coeff11, 0.0, 0.0), ), type=ORTHOTROPIC)
        else:
            self.m.materials['Material-2'].Expansion(table=((self.exp_coeff11, 0.0, 0.0), ), type=ORTHOTROPIC)
        self.m.materials['Material-2'].Density(table=((1e-9, ), ))
        self.m.materials['Material-2'].Damping(alpha=alpha)
        self.m.CompositeShellSection(idealization=NO_IDEALIZATION, integrationRule=SIMPSON, layup=(SectionLayer(thickness=(self.thickness*2.0)*self.thickness_ratio, material='Material-1'), SectionLayer(thickness=(self.thickness*2.0)*(1-self.thickness_ratio), material='Material-2')), name='Section-1', poissonDefinition=DEFAULT, preIntegrate=OFF, symmetric=False, temperature=GRADIENT, thicknessModulus=None, thicknessType=UNIFORM, useDensity=OFF)
        self.m.CompositeShellSection(idealization=NO_IDEALIZATION, integrationRule=SIMPSON, layup=(SectionLayer(thickness=(self.thickness*2.0)*self.thickness_ratio, material='Material-2'), SectionLayer(thickness=(self.thickness*2.0)*(1-self.thickness_ratio), material='Material-1')), name='Section-2', poissonDefinition=DEFAULT, preIntegrate=OFF, symmetric=False, temperature=GRADIENT, thicknessModulus=None, thicknessType=UNIFORM, useDensity=OFF)
        self.Part_1.SectionAssignment(offset=0.0, offsetField='', offsetType=MIDDLE_SURFACE, region=self.Part_1.sets['Set-1'], sectionName='Section-1', thicknessAssignment=FROM_SECTION)
        self.Part_1.SectionAssignment(offset=0.0, offsetField='', offsetType=MIDDLE_SURFACE, region=self.Part_1.sets['Set-2'], sectionName='Section-2', thicknessAssignment=FROM_SECTION)

    def create_assembly(self):
        self.name_Assem  = 'Part-1-1'
        self.m.rootAssembly.DatumCsysByDefault(CARTESIAN)
        self.m.rootAssembly.Instance(dependent=ON, name='Part-1-1', part=self.Part_1)

    def mesh_part(self):
        mdb.models[self.model].parts['Part-1'].setMeshControls(elemShape=QUAD_DOMINATED, regions=mdb.models[self.model].parts['Part-1'].faces, algorithm=ADVANCING_FRONT, allowMapped=True)
        self.Part_1.setElementType(elemTypes=(ElemType(elemCode=S4R, elemLibrary=STANDARD, secondOrderAccuracy=OFF, hourglassControl=DEFAULT), ElemType(elemCode=S3, elemLibrary=STANDARD)), regions=(self.Part_1.faces,))
        self.Part_1.seedEdgeBySize(constraint=FINER, deviationFactor=1e-9, edges=self.Part_1.edges, minSizeFactor=1e-9, size=self.mesh_size)
        self.Part_1.generateMesh()

    def apply_point(self,width):
        xrec = [0, self.unitwidth*self.xreps, self.unitwidth*self.xreps, 0]
        yrec = [0, 0, self.unitheight*self.yreps, self.unitheight*self.yreps]

        if self.direction=='X':
            # Find the nodes along left and right side of the part, that are half way up the height of the part +/- width/2
            y_center = self.unitheight*self.yreps/2
            mesh_tol = self.mesh_size/4
            
            # Find nodes on left edge within width range of y_center
            point_A = self.Part_1.Set(name='point_A', nodes=self.Part_1.nodes.getByBoundingBox(
            xrec[0]-mesh_tol, y_center-width/2, -mesh_tol, 
            xrec[0]+mesh_tol, y_center+width/2, mesh_tol))
            
            # Find nodes on right edge within width range of y_center
            point_B = self.Part_1.Set(name='point_B', nodes=self.Part_1.nodes.getByBoundingBox(
            xrec[1]-mesh_tol, y_center-width/2, -mesh_tol, 
            xrec[1]+mesh_tol, y_center+width/2, mesh_tol))

        elif self.direction=='Y':
            # Find the nodes along bottom and top side of the part, that are half way across the width of the part +/- width/2
            x_center = self.unitwidth*self.xreps/2
            mesh_tol = self.mesh_size/4
            
            # Find nodes on bottom edge within width range of x_center
            point_A = self.Part_1.Set(name='point_A', nodes=self.Part_1.nodes.getByBoundingBox(
            x_center-width/2, yrec[0]-mesh_tol, -mesh_tol, 
            x_center+width/2, yrec[0]+mesh_tol, mesh_tol))
            
            # Find nodes on top edge within width range of x_center
            point_B = self.Part_1.Set(name='point_B', nodes=self.Part_1.nodes.getByBoundingBox(
            x_center-width/2, yrec[2]-mesh_tol, -mesh_tol, 
            x_center+width/2, yrec[2]+mesh_tol, mesh_tol))
        xlatticeVec = [self.unitwidth*self.xreps,0]
        ylatticeVec = [0,self.unitheight*self.yreps]
        LatticeVec = (xlatticeVec,ylatticeVec)
        p = self.Part_1
        nS_edges = list((p.sets['edge_bot'].nodes, p.sets['edge_right'].nodes, p.sets['edge_top'].nodes, p.sets['edge_left'].nodes))
        p.Set(nodes=nS_edges, name='edge_nodes')
        nS_corners = list((p.sets['cn_TL'].nodes, p.sets['cn_TR'].nodes, p.sets['cn_BR'].nodes, p.sets['origin'].nodes))
        p.Set(nodes=nS_corners, name='corner_nodes')
        self.NameRef1 = 'REFERENCE-POINT-1'
        self.NameRef2 = 'REFERENCE-POINT-2'
        generate_RP(self.m, THREE_D,self.NameRef1, self.NameRef2)
        
        # Create equation constraints to tie reference points to point sets
        # Get node labels for point_A and point_B
        point_A_nodes = list(self.Part_1.sets['point_A'].nodes)
        point_B_nodes = list(self.Part_1.sets['point_B'].nodes)
   
        for direction in [1,2,3,4,5,6]:
            equation_name = 'Eq-A-U{}'.format(direction)
            self.m.Equation(name=equation_name, terms=(
                (1.0, 'Part-1-1.point_A', direction),
                (-1.0, self.NameRef1, direction)
            ))

            equation_name = 'Eq-B-U{}'.format(direction)
            self.m.Equation(name=equation_name, terms=(
                (1.0, 'Part-1-1.point_B', direction),
                (-1.0, self.NameRef2, direction)
            ))
        self.m.rootAssembly.regenerate()

    def apply_pbc(self):
        xlatticeVec = [self.unitwidth*self.xreps,0]
        ylatticeVec = [0,self.unitheight*self.yreps]
        LatticeVec = (xlatticeVec,ylatticeVec)
        p = self.Part_1
        nS_edges = list((p.sets['edge_bot'].nodes, p.sets['edge_right'].nodes, p.sets['edge_top'].nodes, p.sets['edge_left'].nodes))
        p.Set(nodes=nS_edges, name='edge_nodes')
        nS_corners = list((p.sets['cn_TL'].nodes, p.sets['cn_TR'].nodes, p.sets['cn_BR'].nodes, p.sets['origin'].nodes))
        p.Set(nodes=nS_corners, name='corner_nodes')
        self.NameRef1 = 'REFERENCE-POINT-1'
        self.NameRef2 = 'REFERENCE-POINT-2'
        generate_RP(self.m, THREE_D,self.NameRef1, self.NameRef2)
        self.m.rootAssembly.regenerate()
        PBC_planar(self.m, THREE_D, 'Part-1', xlatticeVec, ylatticeVec, self.NameRef1, self.NameRef2, 'corner_nodes', 'edge_nodes')
        self.m.rootAssembly.rotate(angle=self.angle_part, axisDirection=(0.0, 0.0, 1.0), axisPoint=(0,0,0), instanceList=(self.name_Assem, ))
        self.m.rootAssembly.regenerate()

    def create_step(self, step_name='Step-1', timePeriod=1):
        self.m.ImplicitDynamicsStep(alpha=DEFAULT, amplitude=RAMP, application=QUASI_STATIC, initialConditions=OFF, initialInc=0.5, maxNumInc=10000, minInc=5e-15, name=step_name, nlgeom=ON, nohaf=OFF, previous='Initial', timePeriod=timePeriod)
        self.m.Temperature(createStepName='Initial', crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=UNIFORM, magnitudes=(0.0, ), name='Predefined Field-1', region=self.m.rootAssembly.instances['Part-1-1'].sets['All'])
        
        # Setup amplitude and temperature field  
        amplitude(self.m,'Part-1',self.NameRef1,self.NameRef2,'Amp-1',50,step_name=step_name, exp_fact = 1,num_odb_TP = 200,num_csv_TP=200)
        self.m.TabularAmplitude(name='Amp-1', timeSpan=STEP, smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (1.0, 1.0)))
        self.m.Temperature(amplitude='Amp-1', createStepName=step_name, crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=UNIFORM, magnitudes=(1.0, ), name='Predefined Field-2', region=self.m.rootAssembly.instances['Part-1-1'].sets['All'])
    
    def create_step_buckle(self, step_name='Step-1', timePeriod=1):
        self.buckle_step = True
        # self.m.BuckleStep(blockSize=DEFAULT, eigensolver=LANCZOS, 
        #     maxBlocks=DEFAULT, minEigen=None, name=step_name, numEigen=10, previous=
        #     'Initial')
        self.m.BuckleStep(maxIterations=200, name=step_name, numEigen=5, 
                previous='Initial', vectors=100)
        self.m.Temperature(createStepName='Initial', crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=UNIFORM, magnitudes=(0.0, ), name='Predefined Field-1', region=self.m.rootAssembly.instances['Part-1-1'].sets['All'])
        
        # Setup amplitude and temperature field  
        self.m.Temperature(createStepName=step_name, crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=UNIFORM, magnitudes=(1.0, ), name='Predefined Field-2', region=self.m.rootAssembly.instances['Part-1-1'].sets['All'])

    def apply_bc_buckle(self,deactivate=False):
        REF1 = self.m.rootAssembly.sets[self.NameRef1]
        REF2 = self.m.rootAssembly.sets[self.NameRef2]
        DeformationF=[(1.0, UNSET), (UNSET, UNSET)]
        self.m.DisplacementBC(amplitude=UNSET, createStepName='Step-1', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='BC-REF1_2', region=REF1, u1=UNSET, u2=0, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET)
        self.m.DisplacementBC(amplitude=UNSET, createStepName='Step-1', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='BC-REF2_2', region=REF2, u1=0, u2=UNSET, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET)

        # Deactivate these boundary conditions in Step-2
        if deactivate:
            self.m.boundaryConditions['BC-REF1_2'].deactivate('Step-2')
            self.m.boundaryConditions['BC-REF2_2'].deactivate('Step-2')
        self.m.DisplacementBC(amplitude=UNSET, createStepName='Step-1', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='BC-ORIGIN', region=self.m.rootAssembly.sets['Part-1-1.origin'], u1=0, u2=UNSET, u3=0, ur1=UNSET, ur2=UNSET, ur3=UNSET)

    def apply_bc_buckle_point(self):
        REF1 = self.m.rootAssembly.sets[self.NameRef1]
        REF2 = self.m.rootAssembly.sets[self.NameRef2]
        DeformationF=[(1.0, UNSET), (UNSET, UNSET)]
        self.m.DisplacementBC(amplitude='Amp-1', createStepName='Step-1', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='BC-REF1_2', region=REF1, u1=0, u2=0, u3=0, ur1=UNSET, ur2=UNSET, ur3=UNSET)
        if self.direction == 'X':
            self.m.DisplacementBC(amplitude='Amp-1', createStepName='Step-1', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='BC-REF2_2', region=REF2, u1=UNSET, u2=0, u3=0, ur1=UNSET, ur2=UNSET, ur3=UNSET)
        elif self.direction == 'Y':
            self.m.DisplacementBC(amplitude='Amp-2', createStepName='Step-1', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='BC-REF2_2', region=REF2, u1=0, u2=UNSET, u3=0, ur1=UNSET, ur2=UNSET, ur3=UNSET)
        self.m.DisplacementBC(amplitude=UNSET, createStepName='Step-1', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='BC-ORIGIN', region=self.m.rootAssembly.sets['Part-1-1.origin'], u1=0, u2=UNSET, u3=0, ur1=UNSET, ur2=UNSET, ur3=UNSET)

    def apply_bc_point(self,deformation,width):
        REF1 = self.m.rootAssembly.sets[self.NameRef1]
        REF2 = self.m.rootAssembly.sets[self.NameRef2]
        xrec = [0, self.unitwidth*self.xreps, self.unitwidth*self.xreps, 0]
        yrec = [0, 0, self.unitheight*self.yreps, self.unitheight*self.yreps]

        if self.direction=='X':
            # Find the nodes along left and right side of the part, that are half way up the height of the part +/- width/2
            y_center = self.unitheight*self.yreps/2
            mesh_tol = self.mesh_size/4
            
            # Find nodes on left edge within width range of y_center
            point_A = self.Part_1.Set(name='point_A', nodes=self.Part_1.nodes.getByBoundingBox(
            xrec[0]-mesh_tol, y_center-width/2, -mesh_tol, 
            xrec[0]+mesh_tol, y_center+width/2, mesh_tol))
            
            # Find nodes on right edge within width range of y_center
            point_B = self.Part_1.Set(name='point_B', nodes=self.Part_1.nodes.getByBoundingBox(
            xrec[1]-mesh_tol, y_center-width/2, -mesh_tol, 
            xrec[1]+mesh_tol, y_center+width/2, mesh_tol))

        elif self.direction=='Y':
            # Find the nodes along bottom and top side of the part, that are half way across the width of the part +/- width/2
            x_center = self.unitwidth*self.xreps/2
            mesh_tol = self.mesh_size/4
            
            # Find nodes on bottom edge within width range of x_center
            point_A = self.Part_1.Set(name='point_A', nodes=self.Part_1.nodes.getByBoundingBox(
            x_center-width/2, yrec[0]-mesh_tol, -mesh_tol, 
            x_center+width/2, yrec[0]+mesh_tol, mesh_tol))
            
            # Find nodes on top edge within width range of x_center
            point_B = self.Part_1.Set(name='point_B', nodes=self.Part_1.nodes.getByBoundingBox(
            x_center-width/2, yrec[2]-mesh_tol, -mesh_tol, 
            x_center+width/2, yrec[2]+mesh_tol, mesh_tol))
            

        else:
            print("Axis not recognized, please use 'X' or 'Y'")


        if self.direction == 'X':
            # For X-axis loading: fix point_A (left side), pull point_B (right side) in X direction
            # For Y-axis loading: fix point_A (bottom side), pull point_B (top side) in Y direction
            self.m.DisplacementBC(amplitude='Amp-2', createStepName='Step-1', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='BC-REF1_2', region=REF1, u1=0, u2=0, u3=0, ur1=UNSET, ur2=UNSET, ur3=UNSET)
            self.m.DisplacementBC(amplitude='Amp-2', createStepName='Step-1', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='BC-REF2_2', region=REF2, u1=1, u2=0, u3=0, ur1=UNSET, ur2=UNSET, ur3=UNSET)
        elif self.direction == 'Y':
            # For Y-axis loading: fix point_A (bottom side), pull point_B (top side) in Y direction
            self.m.DisplacementBC(amplitude='Amp-2', createStepName='Step-1', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='BC-REF1_2', region=REF1, u1=0, u2=0, u3=0, ur1=UNSET, ur2=UNSET, ur3=UNSET)
            self.m.DisplacementBC(amplitude='Amp-2', createStepName='Step-1', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='BC-REF2_2', region=REF2, u1=0, u2=1, u3=0, ur1=UNSET, ur2=UNSET, ur3=UNSET)

    def apply_bc_deformation(self):
        REF1 = self.m.rootAssembly.sets[self.NameRef1]
        REF2 = self.m.rootAssembly.sets[self.NameRef2]
        DeformationF=[(1.0, UNSET), (UNSET, UNSET)]
        self.m.DisplacementBC(amplitude='Amp-2', createStepName='Step-1', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='BC-REF1_2', region=REF1, u1=DeformationF[0][0], u2=DeformationF[0][1], u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET)
        self.m.DisplacementBC(amplitude='Amp-2', createStepName='Step-1', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='BC-REF2_2', region=REF2, u1=DeformationF[1][0], u2=DeformationF[1][1], u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET)
        self.m.DisplacementBC(amplitude=UNSET, createStepName='Step-1', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='BC-ORIGIN', region=self.m.rootAssembly.sets['Part-1-1.origin'], u1=0, u2=UNSET, u3=0, ur1=UNSET, ur2=UNSET, ur3=UNSET)

    def create_job(self, job_name, buckle=True,numCpus = 1,res=False):
        mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF, explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF, memory=90, memoryUnits=PERCENTAGE, model=self.model, modelPrint=OFF, multiprocessingMode=DEFAULT, name=job_name, nodalOutputPrecision=SINGLE, numCpus=numCpus, numGPUs=0, parallelizationMethodExplicit=DOMAIN, numDomains=numCpus, activateLoadBalancing=False, queue=None, resultsFormat=ODB, scratch='', type=ANALYSIS, waitHours=0, waitMinutes=0)

        mdb.jobs[job_name].submit()
        mdb.jobs[job_name].waitForCompletion()

    def post_process(self, job_name):
        project = job_name
        odb = openOdb(project + '.odb')
        session.viewports['Viewport: 1'].setValues(displayedObject=odb)
        if self.direction == 'X':
            F11_R=session.XYDataFromHistory(name='F11',odb=odb,outputVariableName='Spatial displacement: U1 PI: '+self.NameRef1.upper()+' Node 1 in NSET '+self.NameRef1.upper())
        elif self.direction == 'Y':
            F11_R=session.XYDataFromHistory(name='F11',odb=odb,outputVariableName='Spatial displacement: U2 PI: '+self.NameRef1.upper()+' Node 1 in NSET '+self.NameRef2.upper())

        if '_buckle' in job_name:
            # For buckle job, process F11 data and create Amp-2 for next step
            F11_tuple = ()
            for i in range(0,len(F11_R)):
                F11_tuple = F11_tuple + ((F11_R[i][0], F11_R[i][1]),)
            F11_tuple = F11_tuple + ((self.simulation_time, 0.5),)
            self.m.TabularAmplitude(name='Amp-2', timeSpan=STEP, smooth=SOLVER_DEFAULT, data=F11_tuple)
            
            # Setup time points and history output for strain energy
            time_points = [(1.0 + i * (self.simulation_time - 1.0) / 199.0,) for i in range(200)]
            self.m.TimePoint(name='TimePoints-SE', points=tuple(time_points))
            self.m.TimePoint(name='TimePoints-CSV', points=tuple(time_points))
            time_points = [(i * self.simulation_time / 399.0,) for i in range(400)]
            self.m.TimePoint(name='TimePoints-ODB', points=tuple(time_points))
            self.m.HistoryOutputRequest(createStepName='Step-1', name='H-Output-SE', timePoint='TimePoints-SE', variables=('ALLSE', ))
        else:
            # For main job, extract and save F11 and SE data
            SE_R=session.XYDataFromHistory(name='SE',odb=odb,outputVariableName='Strain energy: ALLSE for Whole Model (Repeated: key = Step-1, 36)')
            F11_list = []
            SE_list = []
            for i in range(0,len(F11_R)):
                F11_list.append(F11_R[i][1])
                SE_list.append((SE_R[i][1]))
            
            # Combine F11_list and SE_list into a 2D array
            combined_data = np.column_stack((F11_list, SE_list))
            np.savetxt(job_name+'_F11_SE.csv', combined_data, delimiter=',', header='F11,SE', comments='')
        
        odb.close()

    def post_process_point(self, job_name):
        project = job_name
        odb = openOdb(project + '.odb')
        session.viewports['Viewport: 1'].setValues(displayedObject=odb)

        # For buckle job, process F11 data and create Amp-2 for next step
        F11_tuple = ()
        F11_tuple = F11_tuple + ((0.0, 0),)
        F11_tuple = F11_tuple + ((1.0, 0),)
        F11_tuple = F11_tuple + ((self.simulation_time, self.deformation),)
        # F11_tuple = F11_tuple + ((3.0, self.deformation),)
        # F11_tuple = F11_tuple + ((4.0, self.deformation*0.85),)
        self.m.TabularAmplitude(name='Amp-2', timeSpan=STEP, smooth=SOLVER_DEFAULT, data=F11_tuple)
      
        # Setup time points and history output for strain energy
        time_points = [(1.0 + i * (self.simulation_time - 1.0) / 199.0,) for i in range(200)]
        self.m.TimePoint(name='TimePoints-SE', points=tuple(time_points))
        self.m.TimePoint(name='TimePoints-CSV', points=tuple(time_points))
        time_points = [(i * self.simulation_time / 199.0,) for i in range(200)]
        self.m.TimePoint(name='TimePoints-ODB', points=tuple(time_points))
        self.m.HistoryOutputRequest(createStepName='Step-1', name='H-Output-SE', timePoint='TimePoints-SE', variables=('ALLSE', ))

        odb.close()
        return F11_tuple
    
    def post_process2(self, job_name):
        project = job_name
        odb = openOdb(project + '.odb')
        session.viewports['Viewport: 1'].setValues(displayedObject=odb)
        if self.direction == 'X':
            F11_R=session.XYDataFromHistory(name='F11',odb=odb,outputVariableName='Spatial displacement: U1 PI: '+self.NameRef1.upper()+' Node 1 in NSET '+self.NameRef1.upper())
            F22_R=session.XYDataFromHistory(name='F11',odb=odb,outputVariableName='Spatial displacement: U1 PI: '+self.NameRef2.upper()+' Node 1 in NSET '+self.NameRef2.upper())
        elif self.direction == 'Y':
            F11_R=session.XYDataFromHistory(name='F22',odb=odb,outputVariableName='Spatial displacement: U2 PI: '+self.NameRef1.upper()+' Node 1 in NSET '+self.NameRef1.upper())
            F22_R=session.XYDataFromHistory(name='F22',odb=odb,outputVariableName='Spatial displacement: U2 PI: '+self.NameRef2.upper()+' Node 1 in NSET '+self.NameRef2.upper())

        # For main job, extract and save F11 and SE data
        SE_R=session.XYDataFromHistory(name='SE',odb=odb,outputVariableName='Strain energy: ALLSE for Whole Model (Repeated: key = Step-1, 36)')
        F11_list = []
        SE_list = []
        for i in range(0,len(F11_R)):
            F11_list.append(np.abs(F11_R[i][1]) + np.abs(F22_R[i][1]))
            SE_list.append((SE_R[i][1]))
        
        # Combine F11_list and SE_list into a 2D array
        combined_data = np.column_stack((F11_list, SE_list))
        np.savetxt('strain_energy_displacement.csv', combined_data.astype(float), delimiter=',', header='Displacement (mm),Strain Energy (mJ)', comments='')
        
        odb.close()

    def run_point(self):        
        Mdb()

        self.setup_directory()
        # First simulation: Initial buckling
        self.create_part()
        self.assign_material(alpha=4.0)
        self.create_assembly()
        self.mesh_part()
        self.apply_pbc()
        self.create_step(step_name='Step-1', timePeriod=1)
        # self.m.ImplicitDynamicsStep(alpha=DEFAULT, amplitude=RAMP, application=QUASI_STATIC, initialConditions=OFF, initialInc=0.5, maxNumInc=10000, minInc=5e-15, name='Step-2', nlgeom=ON, nohaf=OFF, previous='Step-1', timePeriod=1)
        self.apply_bc_buckle(deactivate=False)
        self.m.steps['Step-1'].Restart(frequency=0, numberIntervals=1, 
            overlay=ON, timeMarks=OFF)
        width = self.a/2
        xrec = [0, self.unitwidth*self.xreps, self.unitwidth*self.xreps, 0]
        yrec = [0, 0, self.unitheight*self.yreps, self.unitheight*self.yreps]

        if self.direction=='X':
            # Find the nodes along left and right side of the part, that are half way up the height of the part +/- width/2
            y_center = self.unitheight*self.yreps/2
            mesh_tol = self.mesh_size/4
            
            # Find nodes on left edge within width range of y_center
            point_A = self.Part_1.Set(name='point_A', nodes=self.Part_1.nodes.getByBoundingBox(
            xrec[0]-mesh_tol, y_center-width/2, -mesh_tol, 
            xrec[0]+mesh_tol, y_center+width/2, mesh_tol))
            
            # Find nodes on right edge within width range of y_center
            point_B = self.Part_1.Set(name='point_B', nodes=self.Part_1.nodes.getByBoundingBox(
            xrec[1]-mesh_tol, y_center-width/2, -mesh_tol, 
            xrec[1]+mesh_tol, y_center+width/2, mesh_tol))

        elif self.direction=='Y':
            # Find the nodes along bottom and top side of the part, that are half way across the width of the part +/- width/2
            x_center = self.unitwidth*self.xreps/2
            mesh_tol = self.mesh_size/4
            
            # Find nodes on bottom edge within width range of x_center
            point_A = self.Part_1.Set(name='point_A', nodes=self.Part_1.nodes.getByBoundingBox(
            x_center-width/2, yrec[0]-mesh_tol, -mesh_tol, 
            x_center+width/2, yrec[0]+mesh_tol, mesh_tol))
            
            # Find nodes on top edge within width range of x_center
            point_B = self.Part_1.Set(name='point_B', nodes=self.Part_1.nodes.getByBoundingBox(
            x_center-width/2, yrec[2]-mesh_tol, -mesh_tol, 
            x_center+width/2, yrec[2]+mesh_tol, mesh_tol))
        
        self.create_job(self.job+'_buckle', buckle=True,res=True,numCpus=5)
        amp2_tuple = self.post_process_point(self.job+'_buckle')

        # First simulation: Initial buckling
        self.create_part()
        self.assign_material(alpha=1400)
        self.create_assembly()
        self.mesh_part()
        self.apply_point(width= self.a/2)
        self.create_step(step_name='Step-1', timePeriod=1)

        # Second simulation: Full deformation with buckle data
        # Recreate step with longer time period for deformation analysis
        # Scale initial and minimum step times based on simulation time
        initial_inc = 0.1 * (self.simulation_time / 2.0)
        min_inc = 5e-12 * (self.simulation_time / 2.0)
        self.m.ImplicitDynamicsStep(alpha=DEFAULT, amplitude=RAMP, application=QUASI_STATIC, initialConditions=OFF, initialInc=initial_inc, maxNumInc=100000, minInc=min_inc, name='Step-1', nlgeom=ON, nohaf=OFF, previous='Initial', timePeriod=self.simulation_time)
        self.m.TabularAmplitude(name='Amp-2', timeSpan=STEP, smooth=SOLVER_DEFAULT, data=amp2_tuple)
        amplitude(self.m,'Part-1',self.NameRef1,self.NameRef2,'Amp-1',50,step_name='Step-1', exp_fact = 1,num_odb_TP = 200,num_csv_TP=200)
        self.m.Temperature(createStepName='Initial', crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, distributionType=UNIFORM, magnitudes=(1.0, ), name='Predefined Field-1', region=self.m.rootAssembly.instances['Part-1-1'].sets['All'])
        
        self.m.InitialState(createStepName='Initial', endIncrement=
            STEP_END, endStep=LAST_STEP, fileName='Job-1_buckle', instances=(
            self.m.rootAssembly.instances['Part-1-1'], ), name=
            'Predefined Field-3', updateReferenceConfiguration=ON)
        
        self.apply_bc_point(deformation=1,width = self.a/2)

        # Setup time points and history output for strain energy
        time_points = [(1.0 + i * (self.simulation_time - 1.0) / 199.0,) for i in range(200)]
        self.m.TimePoint(name='TimePoints-SE', points=tuple(time_points))
        self.m.TimePoint(name='TimePoints-CSV', points=tuple(time_points))
        
        self.m.HistoryOutputRequest(createStepName='Step-1', name=
            'H-Output-SE', timePoint='TimePoints-SE', variables=('ALLSE', ))

        # Regenerate assemblies
        self.m.rootAssembly.regenerate()
        time_points = [(i * 4.0 / 699.0,) for i in range(700)]
        self.m.TimePoint(name='TimePoints-ODB', points=tuple(time_points))
        self.m.steps['Step-1'].control.setValues(allowPropagation=OFF, 
            resetDefaultValues=OFF, timeIncrementation=(4.0, 8.0, 9.0, 16.0, 10.0, 4.0, 
            12.0, 25, 6.0, 3.0, 50.0))

 
        self.create_job(self.job, buckle=False,numCpus=5)
        self.post_process2(self.job)

        # Cleanup and return to original directory
        try:
            for odb_name in session.odbs.keys():
                session.odbs[odb_name].close()
        except:
            pass
        
        os.chdir(self.original_dir)
        session.viewports['Viewport: 1'].setValues(displayedObject=None)
        print("Returned to original directory: {}".format(os.getcwd()))
        print("Simulation complete.")


if __name__ == "__main__":
    # Check current working directory and navigate to buckling analysis folder
    current_dir = os.getcwd()
    
    # Navigate to buckling analysis directory
    while True:
        dir_name = os.path.basename(current_dir)
        
        # Check if we're in buckling analysis directory
        if dir_name == 'Github':
            break
            
        # Check if we've gone too far (reached A7X folder)
        if 'A7X' in dir_name:
            raise RuntimeError("Cannot find 'buckling analysis' directory - reached A7X folder")
            
        # Move up one directory
        parent_dir = os.path.dirname(current_dir)
        
        # Check if we've reached the root
        if parent_dir == current_dir:
            raise RuntimeError("Cannot find 'buckling analysis' directory - reached filesystem root")
            
        current_dir = parent_dir
    
    # Change to the buckling analysis directory
    os.chdir(current_dir)
    
    execfile('abaqus_functions.py')
    # Default parameters dictionary
    default_params = {
        'a': 1.35*4,
        'b': 0.8*12,
        'd': 0.8*12, 
        'thickness': 0.2,
        'thickness_ratio': 0.5,
        'xreps': 4,
        'yreps': 2,
        'exp_coeff11': 0.5*0.35,
        'exp_coeff22': 0.06*0.35,
        'nu': 0.4999,
        'E': 10,  # Will be calculated from thickness if not provided
        'mesh_size': None,  # Will be calculated from a,b,d if not provided
        'pattern': None,  # Pattern configuration
        'direction': None,  # Pulling direction parameter
        'deformation': None,  # Deformation parameter
        'simulation_time': 2.0  # Simulation time for second simulation (default 2.0s)
    }
    
    # You can modify parameters here or pass them as a dictionary
    params = default_params.copy()
    
    # Calculate E if not provided
    if params['E'] is None:
        params['E'] = 0.1/(params['thickness']*2)
    
    # Set parameters
    param_sets = [
        {'a': 1.35*12, 'b': 0.8*12, 'd': 0.8*12,'pattern':'H-shaped','direction':'X','deformation':115,'exp_coeff11':0.165,'exp_coeff22':0.0198,'E':10,'thickness':0.2},
        {'a': 1.35*12, 'b': 0.8*12, 'd': 0.8*12, 'pattern':'Windmill','direction':'Y','deformation':40,'exp_coeff11':0.04,'exp_coeff22':0.165,'E':10},
        {'a': 1.35*12, 'b': 0.8*12, 'd': 0.8*12, 'pattern':'Step','direction':'Y','deformation':65.0,'exp_coeff11':0.04,'exp_coeff22':0.105,'E':10,'thickness':0.2},
    ]
    
    for param_set in param_sets:
        # Update parameters for this run
        current_params = params.copy()
        current_params.update(param_set)
        
        # Recalculate E if thickness changed
        if 'thickness' in param_set and 'E' not in param_set:
            current_params['E'] = 0.1/(current_params['thickness']*2)
        
        print("Running simulation with a={:.1f}, b={:.1f}, d={:.1f}".format(current_params['a']/1.35, current_params['b']/0.8, current_params['d']/0.8))
        sim = BuckleSimulation(**current_params)
        sim.run_point()