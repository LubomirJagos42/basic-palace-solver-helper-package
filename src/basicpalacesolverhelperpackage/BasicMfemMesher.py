##
# author: LuboJ
# descritpion: Basic general mesher to prepare model for MFEM library used by Palace solver
#

import gmsh
import numpy as np
from typing import Literal

class BasicMfemMesher:

    geometryObjectList = []
    internalGeometryObjectIndexCounter = 0
    _meshFieldList = []
    _materialList = {}
    _boundaryConditionList = {}
    _portList = {}
    _lumpedPartList = {}
    _conductivityList = {}

    _gmshGroupIdList = {}
    _gmshGroupIdIndex = 100000
    _gmshGroupIdIndexIncrement = 100000     #after material object, port object or whatever object or group is added internal ID counter increase by this to all groups have different ID

    def __init__(self):
        print("MFEM mesher created")

    def get_tag_after_fragment (self, object_dimtags:list[tuple[int, int]], input_dimtags:list[tuple[int, int]], mapping:list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
        """
        Takes object dimtags from internal geometry manager, input dimtags for whole fragmentation and output mapping and return renewed dimtag list for
        object which was used as input for fragmentation.
        Args:
            object_dimtags: list[tuple[int, int]]: Object dimtag list from internal geometry manager.
            input_dimtags: list[tuple[int, int]]: Input dimtag list which are used for fragmentation.
            mapping: Output mapping after fragmentation, this is used to remap original object dimtags to new ones.

        Returns: list[tuple[int, int]]: Renewed object dimtags after fragmentation.

        """
        resultDimtagList = []
        for objDimtag in object_dimtags:
            wasDimtagFound = False
            for k in range(len(input_dimtags)):
                if objDimtag == input_dimtags[k]:
                    resultDimtagList = resultDimtagList + mapping[k]
                    wasDimtagFound = True
            if wasDimtagFound == False:
                resultDimtagList.append(objDimtag)

        return resultDimtagList

    def setGmshGroupIdIncrement(self, incrementValue:int) -> None:
        self._gmshGroupIdIndexIncrement = incrementValue

    def addStepfile(self, name:str, stepfile:str, priority:int=-1) -> None:
        """
        Import stepfile into internal geometry manager and add its dimtags into it. Set internal geometry type to 'stepfile'
        Args:
            name:str: Object name, under this name it's represented in internal geometry manager.
            stepfile:str: Path to STEP file.
            priority:int: Internal object priority, higher priority number means object is more important and space occupied
                      by it's volume surface is used for meshing over lower priority objects.

        Returns: None
        """

        isObjectAlreadyImported = False
        for importedObject in self.geometryObjectList:
            if "filepath" in importedObject.keys() and importedObject['filepath'] == stepfile:
                isObjectAlreadyImported = True

        if not isObjectAlreadyImported:
            if priority == -1:
                priority = self.internalGeometryObjectIndexCounter
                self.internalGeometryObjectIndexCounter += 1

            self.geometryObjectList.append({"name": name, "dimtags": self.importStepFileAndGetAllNewEntities(stepfile), "priority": priority, "type": "stepfile", "filepath": stepfile})

            self._gmshGroupIdList[name] = self._gmshGroupIdIndex
            self._gmshGroupIdIndex += self._gmshGroupIdIndexIncrement

        return

    def addGmshObjectUsingDimtags(self, name: str, dimtags: list[tuple[int, int]], priority: int = -1, type: Literal["","surface","point","curve","stepfile"] = "") -> None:
        """
        Add object directly into internal geometry manager, dimtags must be specified.
        Args:
            name: str: Object name under which is specified inside internal geometry manager.
            dimtags: list[tuple(int,int)]: Object dimtags.
            priority: int: Priority, higher priority override object with lower priorities.
            type: Literal["","surface","point","curve","stepfile"]: Type of object which should help with other processing.

        Returns: None
        """

        if priority == -1:
            priority = self.internalGeometryObjectIndexCounter
            self.internalGeometryObjectIndexCounter += 1

        self.geometryObjectList.append({"name": name, "dimtags": dimtags, "priority": priority, "type": type})
        self._gmshGroupIdList[name] = self._gmshGroupIdIndex
        self._gmshGroupIdIndex += self._gmshGroupIdIndexIncrement

    def addGmshVolumeObject(self, name, gmshObjectTag, priority=-1):
        _, gmshObjectBoundary = gmsh.model.occ.getSurfaceLoops(gmshObjectTag)
        dimtags = [(3, gmshObjectTag)]
        for tag in gmshObjectBoundary[0]:
            dimtags.append((2, tag))
        self.addGmshObjectUsingDimtags(name, dimtags, priority, "volume")

    def importStepFileAndGetAllNewEntities(self, step_file):

        print(f"Importing {step_file}...")

        # Get entities before import
        gmsh.model.occ.synchronize()
        entities_before = gmsh.model.getEntities()

        # Import
        gmsh.merge(step_file)
        gmsh.model.occ.synchronize()

        # Get new entities
        new_entities = [e for e in gmsh.model.getEntities() if e not in entities_before]

        print(f"  Added {len(new_entities)} tags")

        return new_entities

    def getGeometryObject(self, name) -> dict[str, list[tuple[int, int]], int] | None:
        for geometryObject in self.geometryObjectList:
            if geometryObject['name'] == name:
                return geometryObject

        return None

    def sortGeomtriesBasedOnPriority(self):
        """
        Sort geometries based on their priority from higher to lower priority.
        Returns:

        """
        print("Sort geometry object based on their priority...")
        self.geometryObjectList.sort(key=lambda x: x["priority"], reverse=True)

    def performFragmentationAndReassignTags(self, renewVolumeSurfaces=False):
        print("Performing fragmentation...")
        gmsh.model.occ.synchronize()

        self.sortGeomtriesBasedOnPriority()

        #
        # first do surface fragmentation
        #
        input_dimtags = gmsh.model.occ.getEntities(2)
        out_tags, out_map = gmsh.model.occ.fragment(input_dimtags, [])
        for geometryObject in self.geometryObjectList:
            geometryObject["dimtags"] = self.get_tag_after_fragment(geometryObject["dimtags"], input_dimtags, out_map)

        gmsh.model.occ.synchronize()

        input_dimtags = gmsh.model.occ.getEntities(3)
        if len(input_dimtags) > 0:
            out_tags, out_map = gmsh.model.occ.fragment(input_dimtags, [])
            for geometryObject in self.geometryObjectList:
                geometryObject["dimtags"] = self.get_tag_after_fragment(geometryObject["dimtags"], input_dimtags, out_map)

            gmsh.model.occ.synchronize()

        #
        # do volume fragmentation with surfaces
        #
        input_dimtags = gmsh.model.occ.getEntities(3)
        input_dimtags_tool = gmsh.model.occ.getEntities(2)
        out_tags, out_map = gmsh.model.occ.fragment(input_dimtags, input_dimtags_tool, removeObject=True, removeTool=True)
        for geometryObject in self.geometryObjectList:
            geometryObject["dimtags"] = self.get_tag_after_fragment(geometryObject["dimtags"], input_dimtags + input_dimtags_tool, out_map)

        gmsh.model.occ.synchronize()

        #
        # Find NEW surfaces created outside out_tags
        #   - made with help with ClaudeAI
        #   - there seems to be new surface created after fragmentation between 3D and 2D objects
        #   - THIS IS EXPERIMENTAL SEEMS RUNNING FOR PATCH ANTENNA, NEED TO BE TESTED IN OTHER SCENARIOS
        #
        surfaces_after = set(tag for dim, tag in gmsh.model.occ.getEntities(2))
        out_tags_surfaces = set(tag for dim, tag in out_tags if dim == 2)
        orphan_surfaces = surfaces_after - out_tags_surfaces
        if orphan_surfaces:
            print(f"\n⚠ WARNING: Surfaces created outside out_tags: {orphan_surfaces}")

            # These might be volume boundary surfaces
            # Assign them to appropriate geometry objects based on location/adjacency
            for surf_tag in orphan_surfaces:
                up, down = gmsh.model.getAdjacencies(2, surf_tag)
                print(f"  Surface {surf_tag} touches volumes: {up}")

                for geometryObject in self.geometryList:
                    if hasattr(up, '__len__'):
                        for upItem in up:
                            if (3, upItem) in geometryObject["dimtags"]:
                                geometryObject["dimtags"].append((2, surf_tag))
                    else:
                        if (3, up) in geometryObject["dimtags"]:
                            geometryObject["dimtags"].append((2, surf_tag))

        #
        # synchronize current model
        #
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        #
        # For volume objects remove all 2D surfaces and renew them using gmsh.model.getAdjacencies(...)
        #
        if renewVolumeSurfaces:
            print("Renewing surface dimtags for volumes.")
            for geometryObject in self.geometryObjectList:

                #
                # renew surfaces just for volumes
                #
                if geometryObject["type"] in ["volume", "stepfile"]:

                    #
                    #   remove all 2D dimtags from current volume
                    #
                    for dimtag in geometryObject["dimtags"]:
                        if dimtag[0] == 2:
                            geometryObject["dimtags"].remove(dimtag)

                    #
                    #   get current 2D surfaces for volume and add them to geometries  objects manager
                    #
                    for dimtag in geometryObject["dimtags"]:
                        if dimtag[0] == 3:
                            upward, downward = gmsh.model.getAdjacencies(3, dimtag[1])
                            for tag in downward:
                                geometryObject["dimtags"].append((2, tag))

                    #
                    # TODO: Do also .getAdjacencies() also for 2D surfaces???
                    #

            #
            # Remove duplicate dimtags in lower priority objects
            #
            usedDimtagsInHigherPriorityObjectList = []
            for geometryObject in self.geometryObjectList:
                for dimtag in geometryObject["dimtags"]:
                    if dimtag in usedDimtagsInHigherPriorityObjectList:
                        geometryObject["dimtags"].remove(dimtag)
                    else:
                        usedDimtagsInHigherPriorityObjectList.extend(geometryObject["dimtags"])

        print("Fragmentation finished...")

    def createGroup(self, groupName: str, objectNameOrNameList: str | list[str], dimension, groupTag=-1):
        if type(objectNameOrNameList) == str:
            groupTag = gmsh.model.addPhysicalGroup(dimension, [tag for dim, tag in self.getGeometryObject(objectNameOrNameList)["dimtags"] if dim == dimension], tag=groupTag, name=groupName)
        elif type(objectNameOrNameList) == list:
            geometryDimtagsList = []
            for objectName in objectNameOrNameList:
                geometryDimtagsList.extend(self.getGeometryObject(objectName)["dimtags"])
            groupTag = gmsh.model.addPhysicalGroup(dimension, [tag for dim, tag in geometryDimtagsList if dim == dimension], tag=groupTag, name=groupName)
        else:
            raise("Invalid input for object, it must be str or list of strings!")

        self._gmshGroupIdList[groupName] = groupTag

        return groupTag

    def createGroupsForUntaggedSurfacesAndVolumes(self):
        untaggedSurfaceTagList, untaggedVolumeTagList = self.validate_mesh_attributes()

        groupName = "UNTAGGED_2D"
        gmsh.model.addPhysicalGroup(2, untaggedSurfaceTagList, self._gmshGroupIdIndex, groupName)
        self._gmshGroupIdList[groupName] = self._gmshGroupIdIndex
        self._gmshGroupIdIndex += self._gmshGroupIdIndexIncrement

        groupName = "UNTAGGED_3D"
        gmsh.model.addPhysicalGroup(3, untaggedVolumeTagList, self._gmshGroupIdIndex, groupName)
        self._gmshGroupIdList[groupName] = self._gmshGroupIdIndex
        self._gmshGroupIdIndex += self._gmshGroupIdIndexIncrement

    def cutVolumesInsideModel(self):
        self.sortGeomtriesBasedOnPriority()

        if len(self.geometryObjectList) < 2:
            return

        for k in range(len(self.geometryObjectList)-1):
            toolGeometryObject = self.geometryObjectList[k]

            for m in range(k+1, len(self.geometryObjectList)):
                baseGeometryObject = self.geometryObjectList[m]

                isToolObjectAbleToCut = False
                for dimtag in toolGeometryObject["dimtags"]:
                    if dimtag[0] in [3]:
                        isToolObjectAbleToCut = True

                isBaseObjectAbleToCut = False
                for dimtag in baseGeometryObject["dimtags"]:
                    if dimtag[0] in [2,3]:
                        isBaseObjectAbleToCut = True

                if (isToolObjectAbleToCut and isBaseObjectAbleToCut) == False:
                    continue

                #   Without this cutting all 3D objects between each other there is error in palace from MFEM library:
                #         Verification failed: (faces_info[gf].Elem2No < 0) is False:
                #          --> Invalid mesh topology.  Interior triangular face found connecting elements 20, 21 and 40.
                #          ... in function: void mfem::Mesh::AddTriangleFaceElement(int, int, int, int, int, int)
                #          ... in file: /opt/palace-build/extern/mfem/mesh/mesh.cpp:8198
                #   This error can be replicate just by loading mesh in python mfem library.

                print(f"Cutting '{toolGeometryObject["name"]}' from '{baseGeometryObject["name"]}'...")
                try:
                    baseDimtags = [ (gmshTuple[0], gmshTuple[1]) for gmshTuple in baseGeometryObject["dimtags"] if gmshTuple[0] in [2,3]]
                    toolDimtags = [(gmshTuple[0], gmshTuple[1]) for gmshTuple in toolGeometryObject["dimtags"] if gmshTuple[0] in [2,3]]
                    outDimtags, outDimtagsMap = gmsh.model.occ.cut(
                        baseDimtags,
                        toolDimtags,
                        removeObject=True,
                        removeTool=False
                    )
                    self.get_tag_after_fragment(baseGeometryObject["dimtags"], baseDimtags, outDimtagsMap)
                    gmsh.model.occ.synchronize()
                except Exception as e:
                    print(e)
                    pass

    def cutOnlyVolumesInModelBetweenEachOther(self, allowSurfacesToBeCutted=False):
        self.sortGeomtriesBasedOnPriority()

        if len(self.geometryObjectList) < 2:
            return

        for k in range(len(self.geometryObjectList)-1):
            toolGeometryObject = self.geometryObjectList[k]

            for m in range(k+1, len(self.geometryObjectList)):
                baseGeometryObject = self.geometryObjectList[m]

                isToolObjectAbleToCut = False
                for dimtag in toolGeometryObject["dimtags"]:
                    if dimtag[0] in [3]:
                        isToolObjectAbleToCut = True

                isBaseObjectAbleToCut = False
                baseObjectsAllowedDiemnsionToBeUsed = [2,3] if allowSurfacesToBeCutted else [3]
                for dimtag in baseGeometryObject["dimtags"]:
                    if dimtag[0] in baseObjectsAllowedDiemnsionToBeUsed:
                        isBaseObjectAbleToCut = True

                if (isToolObjectAbleToCut and isBaseObjectAbleToCut) == False:
                    continue

                #   Without this cutting all 3D objects between each other there is error in palace from MFEM library:
                #         Verification failed: (faces_info[gf].Elem2No < 0) is False:
                #          --> Invalid mesh topology.  Interior triangular face found connecting elements 20, 21 and 40.
                #          ... in function: void mfem::Mesh::AddTriangleFaceElement(int, int, int, int, int, int)
                #          ... in file: /opt/palace-build/extern/mfem/mesh/mesh.cpp:8198
                #   This error can be replicate just by loading mesh in python mfem library.

                print(f"Cutting '{toolGeometryObject["name"]}' from '{baseGeometryObject["name"]}'...")
                try:
                    baseDimtags = [(gmshTuple[0], gmshTuple[1]) for gmshTuple in baseGeometryObject["dimtags"] if gmshTuple[0] in baseObjectsAllowedDiemnsionToBeUsed]
                    toolDimtags = [(gmshTuple[0], gmshTuple[1]) for gmshTuple in toolGeometryObject["dimtags"] if gmshTuple[0] in [3]]
                    outDimtags, outDimtagsMap = gmsh.model.occ.cut(
                        baseDimtags,
                        toolDimtags,
                        removeObject=True,
                        removeTool=False
                    )
                    gmsh.model.occ.synchronize()
                    self.get_tag_after_fragment(baseGeometryObject["dimtags"], baseDimtags, outDimtagsMap)
                    gmsh.model.occ.synchronize()
                except Exception as e:
                    print(e)
                    pass

    def addMeshFieldToList(self, fieldObj):
        self._meshFieldList.append(fieldObj)

    def setSurfaceMeshSize(self, geometryObjectNameOrList: str | list[str], sizeMin: float=0.0, sizeMax: float=0.0, distanceMin: float=0.0, distanceMax: float=0.0, useDistanceFrom: list[Literal["edges", "surface", "points"]] = ["edges"]) -> int:
        """
        Specify surface mesh using field.
        Args:
            geometryObjectNameOrList:
            sizeMin: Minimal mesh size, this parameters specified mesh size and is main purpose of this method
            sizeMax: Mesh size outsize maximum distance from specified objects from which are mesh distance calculated
            distanceMin: Inside this distance mesh is fine (sizeMin)
            distanceMax: Outside this distance mesh is coarse (sizeMax)
            useDistanceFrom:Literal["edges", "surface", "points"]: Mesh restricted size would be calculated by distance from specified object, by default it calculate distance from "edges"

        Returns: field tag
        """

        all_surface_tags = []
        if type(geometryObjectNameOrList) == str:
            all_surface_tags.extend([tag for dim, tag in self.getGeometryObject(geometryObjectNameOrList)["dimtags"] if dim == 2])
        else:
            for geometryObjectName in geometryObjectNameOrList:
                all_surface_tags.extend([tag for dim, tag in self.getGeometryObject(geometryObjectName)["dimtags"] if dim == 2])

        # Simple distance-based field
        field_dist = gmsh.model.mesh.field.add("Distance")
        if "surface" in useDistanceFrom:
            gmsh.model.mesh.field.setNumbers(field_dist, "SurfacesList", all_surface_tags)
        if "edges" in useDistanceFrom:
            all_edges = gmsh.model.getBoundary([(2, surfaceTag) for surfaceTag in all_surface_tags], combined=False, recursive=False, oriented=False)
            all_edges = [dimtag[1] for dimtag in all_edges]
            gmsh.model.mesh.field.setNumbers(field_dist, "CurvesList", all_edges)
        if "points" in useDistanceFrom:
            all_edges = gmsh.model.getBoundary([(2, surfaceTag) for surfaceTag in all_surface_tags], combined=False, recursive=False, oriented=False)
            all_points = gmsh.model.getBoundary(all_edges, combined=False, recursive=False, oriented=False)
            all_points = [dimtag[1] for dimtag in all_points]
            gmsh.model.mesh.field.setNumbers(field_dist, "PointsList", all_points)

        field_threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field_threshold, "InField", field_dist)
        gmsh.model.mesh.field.setNumber(field_threshold, "SizeMin", sizeMin)  # Fine near surfaces
        gmsh.model.mesh.field.setNumber(field_threshold, "SizeMax", sizeMax)  # Coarse far away
        gmsh.model.mesh.field.setNumber(field_threshold, "DistMin", distanceMin)
        gmsh.model.mesh.field.setNumber(field_threshold, "DistMax", distanceMax)

        self._meshFieldList.append(field_threshold)

        return field_threshold

    def setPointMeshSize(self, geometryObjectNameOrList: str | list[str], sizeMin: float=0.0, sizeMax: float=0.0, distanceMin: float=0.0, distanceMax: float=0.0) -> int:
        """
        Specified mesh size around gmsh object points. Object is specified by its name in internal geometry object manager.
        Args:
            geometryObjectNameOrList:
            sizeMin: Fine mesh size inside distance min.
            sizeMax: Coarse mesh size outside distance max.
            distanceMin: Inside this distance object mesh is fine mesh.
            distanceMax: Outside this distance object is set to be coarse.

        Returns: field tag
        """

        all_point_tags = []
        if type(geometryObjectNameOrList) == str:
            all_point_tags.extend([tag for dim, tag in self.getGeometryObject(geometryObjectNameOrList)["dimtags"] if dim == 0])
        else:
            for geometryObjectName in geometryObjectNameOrList:
                all_point_tags.extend([tag for dim, tag in self.getGeometryObject(geometryObjectName)["dimtags"] if dim == 0])

        # Simple distance-based field
        field_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field_dist, "PointsList", all_point_tags)

        field_threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field_threshold, "InField", field_dist)
        gmsh.model.mesh.field.setNumber(field_threshold, "SizeMin", sizeMin)  # Fine near points
        gmsh.model.mesh.field.setNumber(field_threshold, "SizeMax", sizeMax)  # Coarse far away
        gmsh.model.mesh.field.setNumber(field_threshold, "DistMin", distanceMin)
        gmsh.model.mesh.field.setNumber(field_threshold, "DistMax", distanceMax)

        self._meshFieldList.append(field_threshold)

        return field_threshold

    def setCurveMeshSizeForDimtags(self, geometryObjectNameOrDimtagList: list[tuple[int,int]], sizeMin: float=0.0, sizeMax: float=1e22, distanceMin: float=0.0, distanceMax: float=1e22) -> int:
        """
        Set fine mesh size around curves in distance closer than distanceMin, if internal geometry manager name provided as dimtags are used edge
        dimtags extracted from object.
        Args:
            geometryObjectNameOrDimtagList:
            sizeMin: Fine mesh size.
            sizeMax: By default is set to huge number to have affect to infinity i.e. ignore this field as it's supposed that background field for whole model is set to 'Min'
            distanceMin: Distance around curve where fine mesh is applied
            distanceMax: ignored field therefore this field is set to huge number 1e22 to exclude sizeMax be aplicable

        Returns: int: field number or -1 if no field was created
        """

        all_curve_tags = []
        if type(geometryObjectNameOrDimtagList) == str:
            all_curve_tags = [tag for dim, tag in self.getGeometryObjectEdges(geometryObjectNameOrDimtagList) if dim == 1]
        elif type(geometryObjectNameOrDimtagList) == list:
            all_curve_tags = [tag for dim, tag in geometryObjectNameOrDimtagList if dim == 1]

        if len(all_curve_tags) > 0:
            # Simple distance-based field
            field_dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(field_dist, "CurvesList", all_curve_tags)

            field_threshold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(field_threshold, "InField", field_dist)
            gmsh.model.mesh.field.setNumber(field_threshold, "SizeMin", sizeMin)  # Fine near points
            gmsh.model.mesh.field.setNumber(field_threshold, "SizeMax", sizeMax)  # Coarse far away
            gmsh.model.mesh.field.setNumber(field_threshold, "DistMin", distanceMin)
            gmsh.model.mesh.field.setNumber(field_threshold, "DistMax", distanceMax)

            self._meshFieldList.append(field_threshold)

            return field_threshold

        return -1

    def setSizeOnEdge(self, objectName: str = "", tags: list[int] | list[tuple[int,int]] = [], max_size: float = 0.0, out_size: float | None = None) -> None:
        """Define the size of the mesh on an edge

        Args:
            tags (list[int] | list[tuple[int,int]]): The tags or dimtags of the geometry
            max_size (float): The maximum size (in meters)
            out_size (float): Size outside (in meters)
        """
        if len(tags) > 0:
            if type(tags[0]) == tuple:
                tags = [dimtag[1] for dimtag in tags if dimtag[0] == 1] #extract just curves tags

        if len(objectName) > 0:
            tags.extend([tag for dim, tag in self.getGeometryObjectEdges(objectName)])

        constantTag = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.set_numbers(constantTag, "CurvesList", tags)
        gmsh.model.mesh.field.set_number(constantTag, "VIn", max_size)
        if out_size is not None:
            gmsh.model.mesh.field.set_number(constantTag, "VOut", out_size)
        self._meshFieldList.append(constantTag)

    def setSizeOnFace(self, geometryObjectNameOrList: str | list[str], max_size: float=0.0):
        # Collect all surface tags
        all_surfaces = []
        if type(geometryObjectNameOrList) == str:
            all_surfaces.extend([tag for dim, tag in self.getGeometryObject(geometryObjectNameOrList)["dimtags"] if dim == 2])
        else:
            for geometryObjectName in geometryObjectNameOrList:
                all_surfaces.extend([tag for dim, tag in self.getGeometryObject(geometryObjectName)["dimtags"] if dim == 2])

        constantTag = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.set_numbers(constantTag, "SurfacesList", all_surfaces)
        gmsh.model.mesh.field.set_number(constantTag, "VIn", max_size)

        self._meshFieldList.append(constantTag)

        return constantTag

    def setSizeForVolume(self, geometryObjectNameOrList: str | list[str], max_size: float=0.0):
        # Collect all surface tags
        all_volumes = []
        if type(geometryObjectNameOrList) == str:
            all_volumes.extend([tag for dim, tag in self.getGeometryObject(geometryObjectNameOrList)["dimtags"] if dim == 3])
        else:
            for geometryObjectName in geometryObjectNameOrList:
                all_volumes.extend([tag for dim, tag in self.getGeometryObject(geometryObjectName)["dimtags"] if dim == 3])

        constantTag = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.set_numbers(constantTag, "VolumesList", all_volumes)
        gmsh.model.mesh.field.set_number(constantTag, "VIn", max_size)

        self._meshFieldList.append(constantTag)

        return constantTag

    def setSizeBoundary(self,
                          boundaryObjectName: str,
                          size: float,
                          max_size: float | None = None,
                          growth_rate: float = 3) -> None:

        """Refine the mesh size along the boundary of a conducting surface

        The growth rate determines how quickly the mesh size is allowed to increase away from the face boundary.

        Args:
            boundary (str): Name of boundary/surface object to refine the mesh on
            size (float): The mesh size limit in meters
            growth_rate (float, optional): The mesh growth rate. Defaults to 3.
            max_size (float, optional): The maximum mesh size. Defaults to None.
        """
        dimtagsList = self.getGeometryObject(boundaryObjectName)["dimtags"]
        dimtagsList = self.removeDimtagsNotInModel(dimtagsList)

        if max_size is None:
            max_size = size

        growth_distance = (growth_rate * max_size - size) / (growth_rate - 1)
        print(f'Setting boundary size for region {dimtagsList} to {size}, GR={growth_rate}, dist={growth_distance}mm, Max={max_size}mm')

        objectDimension = -1
        for dimtag in dimtagsList:
            if dimtag[0] == 2:
                objectDimension = 2
            if dimtag[0] == 3:
                objectDimension = 3

        if objectDimension > -1:
            nodes = gmsh.model.getBoundary(dimtagsList, combined=False, oriented=False, recursive=False)

            disttag = gmsh.model.mesh.field.add("Distance")
            if objectDimension == 2:
                gmsh.model.mesh.field.setNumbers(disttag, "CurvesList", [n[1] for n in nodes if n[0] == 1])
            if objectDimension == 3:
                gmsh.model.mesh.field.setNumbers(disttag, 'SurfacesList', [n[1] for n in nodes if n[0] == 2])
            gmsh.model.mesh.field.setNumber(disttag, "Sampling", 100)

            thresholdFieldTag = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thresholdFieldTag, "InField", disttag)
            gmsh.model.mesh.field.setNumber(thresholdFieldTag, "SizeMin", size)
            gmsh.model.mesh.field.setNumber(thresholdFieldTag, "SizeMax", max_size)
            gmsh.model.mesh.field.setNumber(thresholdFieldTag, "DistMin", size)
            gmsh.model.mesh.field.setNumber(thresholdFieldTag, "DistMax", growth_distance)

            self.addMeshFieldToList(thresholdFieldTag)

    def setSize(self, objectName: str, size: float, distance: float = 10.0) -> None:
        """
        Set mesh size for point, curve, surface and volume. For surface and volume it's size in it, for curve it's size of mesh on curve and
        for point it's size in distance from point.
        Args:
            objectName: Internal gemoetry manager object name.
            size: Mesh size which will be used.
            distance: For point this is distance from point till where mesh size will be set.
        Returns: None
        """

        objectDimension = self.getGeometryObjectDimension(objectName)
        if objectDimension == 2:
            self.setSizeOnFace(objectName, size)
        elif objectDimension == 3:
            self.setSizeForVolume(objectName, size)
        elif objectDimension == 1:
            self.setSizeOnEdge(objectName=objectName, max_size=size)
        elif objectDimension == 0:
            self.setPointMeshSize(objectName, size, 1e22, distance, 1e22)

    def setBackgroundMinFieldUsingAllDefinedFields(self):
        f_min = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", self._meshFieldList)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    def setBackgroundFieldUsingAllDefinedFields(self, fieldType):
        f_background = gmsh.model.mesh.field.add(fieldType)
        gmsh.model.mesh.field.setNumbers(f_background, "FieldsList", self._meshFieldList)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_background)

    def createGroupsForAllImportedObjects(self, nameList: list[str] = []):
        for geometryObject in self.geometryObjectList:
            geometryObjectName = geometryObject["name"]
            if len(nameList) == 0 or (len(nameList) > 0 and geometryObjectName in nameList):
                self.createGroup(geometryObjectName + "_2D", geometryObjectName, 2, groupTag=self.getGmshGroupId(geometryObjectName))
                self.createGroup(geometryObjectName + "_3D", geometryObjectName, 3, groupTag=self.getGmshGroupId(geometryObjectName)+1)

    def createGroupsForAllMaterials(self):
        for materialName in self._materialList.keys():
            gmshMaterialGroupName = "material_"+materialName
            if gmshMaterialGroupName in self._gmshGroupIdList.keys():
                self.createGroup(gmshMaterialGroupName + "_2D", self._materialList[materialName]["objects"], 2, groupTag=self.getGmshGroupId(gmshMaterialGroupName))
                self.createGroup(gmshMaterialGroupName + "_3D", self._materialList[materialName]["objects"], 3, groupTag=self.getGmshGroupId(gmshMaterialGroupName)+1)

    def createGroupsForAllBoundaryConditions(self):
        for boundaryName in self._boundaryConditionList.keys():
            gmshBoundaryGroupName = "boundary_"+boundaryName
            if gmshBoundaryGroupName in self._gmshGroupIdList.keys():
                self.createGroup(gmshBoundaryGroupName + "_2D", self._boundaryConditionList[boundaryName], 2, groupTag=self.getGmshGroupId(gmshBoundaryGroupName))

    def createGroupsForObjectVolumesUsedInMaterials(self):
        for materialName in self._materialList.keys():
            if "objects" in self._materialList[materialName].keys() and len(self._materialList[materialName]["objects"]) > 0:
                for objectNameAssignedToMaterial in self._materialList[materialName]["objects"]:
                    self.createGroup(objectNameAssignedToMaterial+"_3D", objectNameAssignedToMaterial, 3, groupTag=self.getGmshGroupId(objectNameAssignedToMaterial)+1)

    def createGroupsForObjectSurfacesUsedInBoundaryConditions(self):
        for boundaryName in self._boundaryConditionList.keys():
            for objectNameAssignedToBoundary in self._boundaryConditionList[boundaryName]:

                # if object has no surfaces try to create boundary from its volume tags
                surfaceTagList = [dimtag[1] for dimtag in self.getGeometryObject(objectNameAssignedToBoundary)["dimtags"] if dimtag[0] == 2]
                if len(surfaceTagList) == 0:
                    groupTag = gmsh.model.addPhysicalGroup(2, self.getBoundaryOuter(objectNameAssignedToBoundary), tag=self.getGmshGroupId(objectNameAssignedToBoundary), name=objectNameAssignedToBoundary + "_2D")
                    self._gmshGroupIdList[objectNameAssignedToBoundary + "_2D"] = groupTag
                else:
                    self.createGroup(objectNameAssignedToBoundary + "_2D", objectNameAssignedToBoundary, 2, groupTag=self.getGmshGroupId(objectNameAssignedToBoundary))

        for boundaryName in self._lumpedPartList.keys():
            for objectNameAssignedToBoundary in self._lumpedPartList[boundaryName]["objects"]:

                # if object has no surfaces try to create boundary from its volume tags
                surfaceTagList = [dimtag[1] for dimtag in self.getGeometryObject(objectNameAssignedToBoundary)["dimtags"] if dimtag[0] == 2]
                if len(surfaceTagList) == 0:
                    groupTag = gmsh.model.addPhysicalGroup(2, self.getBoundaryOuter(objectNameAssignedToBoundary), tag=self.getGmshGroupId(objectNameAssignedToBoundary), name=objectNameAssignedToBoundary + "_2D")
                    self._gmshGroupIdList[objectNameAssignedToBoundary + "_2D"] = groupTag
                else:
                    self.createGroup(objectNameAssignedToBoundary + "_2D", objectNameAssignedToBoundary, 2, groupTag=self.getGmshGroupId(objectNameAssignedToBoundary))

    def createGroupsForObjectSurfacesUsedInPort(self):
        # for portObjectName in self._portList.keys():
        for portObj in self._portList.values():
            for portAssignedObjectName in portObj["objects"]:
                self.createGroup(portAssignedObjectName+"_2D", portAssignedObjectName, 2, groupTag=self.getGmshGroupId(portAssignedObjectName))

    def addMaterial(self, name:str="", er:float|None=None, ur:float|None=None, conductivity:float|None=None, tand:float|None=None) -> None:
        if not name in self._materialList.keys():
            self._materialList[name] ={}
            self._gmshGroupIdList["material_"+name] = self._gmshGroupIdIndex
            self._gmshGroupIdIndex += self._gmshGroupIdIndexIncrement

        if er is not None:
            self._materialList[name]["er"] = er
        if ur is not None:
            self._materialList[name]["ur"] = ur
        if conductivity is not None:
            self._materialList[name]["sigma"] = conductivity
        if tand is not None:
            self._materialList[name]["tand"] = tand

        return

    def addObjectToMaterial(self, materialName: str, objectName: str) -> None:
        if not "objects" in self._materialList[materialName].keys():
            self._materialList[materialName]["objects"] = []

        self._materialList[materialName]["objects"].append(objectName)
        self._materialList[materialName]["objects"] = list(set(self._materialList[materialName]["objects"]))

        return

    def addObjectToBoundaryCondition(self, boundaryConditionName: str, objectName: str) -> None:
        if not boundaryConditionName in self._boundaryConditionList.keys():
            self._boundaryConditionList[boundaryConditionName] = []
            self._gmshGroupIdList["boundary_"+boundaryConditionName] = self._gmshGroupIdIndex
            self._gmshGroupIdIndex += self._gmshGroupIdIndexIncrement

        self._boundaryConditionList[boundaryConditionName].append(objectName)
        self._boundaryConditionList[boundaryConditionName] = list(set(self._boundaryConditionList[boundaryConditionName]))

        return

    def getMaterialAttributesAsDictForPalaceSimulationFile(self, materialName:str) -> dict:
        """
        Get material json for material if there are some objects assigned for this material. If no objects are assigned palace sovler will complain
        about missing Attributes.
        Args:
            materialName: Existing material name

        Returns:
            - json material dict
            - None if no objects assigned
        """

        if "objects" in self._materialList[materialName].keys() and len(self._materialList[materialName]["objects"]) > 0:
            materialAttributes = self._materialList[materialName]

            materialObject = {}
            if "er" in materialAttributes.keys():
                materialObject["Permeability"] = materialAttributes["er"]
            if "ur" in materialAttributes.keys():
                materialObject["Permittivity"] = materialAttributes["ur"]
            if "tand" in materialAttributes.keys():
                materialObject["LossTan"] = materialAttributes["tand"]
            if "sigma" in materialAttributes.keys():
                materialObject["Conductivity"] = materialAttributes["sigma"]

            materialObject["Attributes"] = []
            for objectNameAssignedToMaterial in self._materialList[materialName]["objects"]:
                materialObject["Attributes"].append(self._gmshGroupIdList[objectNameAssignedToMaterial+"_3D"])

            return materialObject

        else:
            return None

    def getBoundaryConditionAttributesAsDictForPalaceSimulationFile(self, boundaryName:str) -> dict:
        """
        Create basic simlationConfig["Boundaries"][<boundary name>] object and create just totaly basic object with just "Attributes" property defined.
        If boundary needs to specify more properties like "Impedance" has Rs, Ls, Cs it needs to be added after using this function.
        Args:
            boundaryName:

        Returns:

        """
        boundaryObject = {}

        #
        #   Common property for all boundary condition items, "Attributes" what are gmsh tag to which this boundary condition is assigned
        #
        boundaryObject["Attributes"] = []
        if boundaryName in self._boundaryConditionList.keys():
            for objectNameAssignedToBoundary in self._boundaryConditionList[boundaryName]:
                #assign this boundary condition just to surface, we are using out internal naming convention that imported step file
                #has postfix _2D for its surface
                boundaryObject["Attributes"].append(self._gmshGroupIdList[objectNameAssignedToBoundary + "_2D"])

        if boundaryName in self._lumpedPartList.keys():
            for objectNameAssignedToBoundary in self._lumpedPartList[boundaryName]["objects"]:
                boundaryObject["Attributes"].append(self._gmshGroupIdList[objectNameAssignedToBoundary + "_2D"])

        if boundaryName in self._conductivityList.keys():
            for objectNameAssignedToBoundary in self._conductivityList[boundaryName]["objects"]:
                boundaryObject["Attributes"].append(self._gmshGroupIdList[objectNameAssignedToBoundary + "_2D"])

        return boundaryObject

    def getMaterialNamesList(self):
        return self._materialList.keys()

    def getBoundaryConditionNamesList(self):
        return self._boundaryConditionList.keys()

    def getLumpedPartNamesList(self):
        return self._lumpedPartList.keys()

    def getConductivityNamesList(self):
        return self._conductivityList.keys()

    def getAllMaterialObjectForPalace(self):
        palaceMaterialObject = []
        for materialName in self.getMaterialNamesList():
            materialObject = self.getMaterialAttributesAsDictForPalaceSimulationFile(materialName)
            if not materialObject is None:
                palaceMaterialObject.append(materialObject)

        return palaceMaterialObject

    def getAllBoundaryConditionsObjectForPalace(self) -> dict:
        palaceBoundaryConditionObject = {}

        #
        #   Create JSON definition for boundary condition from boundary conditions
        #       - it takes boundary condition type as key for array which are "Absorbing", "PEC", "PMC", "Ground", "ZeroCharge" and
        #         assign json object to it like {"Attributes": [...]}
        #
        for boundaryName in self.getBoundaryConditionNamesList():
            palaceBoundaryConditionObject[boundaryName] = self.getBoundaryConditionAttributesAsDictForPalaceSimulationFile(boundaryName)

        #
        #   Create JSON definition for boundary condition from lumped parts
        #       - this uses boundary condition "Impedance" in palace which can define parameters Rs, Ls, Cs
        #
        for boundaryName in self.getLumpedPartNamesList():
            if not "Impedance" in palaceBoundaryConditionObject.keys():
                palaceBoundaryConditionObject["Impedance"] = []

            boundaryObject = self.getBoundaryConditionAttributesAsDictForPalaceSimulationFile(boundaryName)
            boundaryObject["Rs"] = self._lumpedPartList[boundaryName]["Rs"]
            boundaryObject["Ls"] = self._lumpedPartList[boundaryName]["Ls"]
            boundaryObject["Cs"] = self._lumpedPartList[boundaryName]["Cs"]
            palaceBoundaryConditionObject["Impedance"].append(boundaryObject)

        #
        #   Add JSON conductivity object
        #
        for boundaryName in self.getConductivityNamesList():
            if not "Conductivity" in palaceBoundaryConditionObject.keys():
                palaceBoundaryConditionObject["Conductivity"] = []

            boundaryObject = self.getBoundaryConditionAttributesAsDictForPalaceSimulationFile(boundaryName)
            boundaryObject["Conductivity"] = self._conductivityList[boundaryName]["Conductivity"]
            boundaryObject["Permeability"] = self._conductivityList[boundaryName]["Permeability"]
            boundaryObject["Thickness"] = self._conductivityList[boundaryName]["Thickness"]
            palaceBoundaryConditionObject["Conductivity"].append(boundaryObject)

        return palaceBoundaryConditionObject

    def getAllLumpedPortObjectForPalace(self):
        palaceLumpedPortObject = []
        for portObj in self._portList.values():
            if portObj["type"] == "lumped":
                #
                # if port is created as:
                #   - plane or whatever 2D structure and its object is type of surface use to identitfy it in gmshIdList just its name
                #   - surface of imported STEP file then use portName+"_2D" since this group will be created and will cover object surface
                #
                gmshPortObjIdList = []
                for objectNameAssignedToPort in portObj["objects"]:
                    portGeometryObjectType = self.getGeometryObject(objectNameAssignedToPort)["type"]
                    gmshPortObjIdList.append(
                        self._gmshGroupIdList[objectNameAssignedToPort]
                        if portGeometryObjectType == "surface"
                        else self._gmshGroupIdList[objectNameAssignedToPort+"_2D"]
                        if portGeometryObjectType == "stepfile"
                        else
                        -1
                    )

                palaceLumpedPortObject.append({
                    "Index": portObj["index"],
                    "Attributes": gmshPortObjIdList,
                    "Direction": portObj["direction"],
                    "R": portObj["R"],
                    "Excitation": True if portObj["excitation"] > 0 else False
                })

        return palaceLumpedPortObject

    def getAllSurfaceCurrentForPortObjectForPalace(self):
        palaceSurfaceCurrentObject = []
        for portObj in self._portList.values():
            if portObj["type"] == "lumped":
                gmshGroupIdList = []
                for objectNameAssignedToPort in portObj["objects"]:
                    gmshGroupIdList.append(self._gmshGroupIdList[objectNameAssignedToPort])

                palaceSurfaceCurrentObject.append({
                    "Index": portObj["index"],
                    "Attributes": gmshGroupIdList,
                    "Direction": portObj["direction"]
                })
        return palaceSurfaceCurrentObject

    def getAllTerminalForPortObjectForPalace(self):
        palaceTerminalObject = []
        for portObj in self._portList.values():
            if portObj["type"] == "lumped":
                gmshGroupIdList = []
                for objectNameAssignedToPort in portObj["objects"]:
                    gmshGroupIdList.append(self._gmshGroupIdList[objectNameAssignedToPort])

                palaceTerminalObject.append({
                    "Index": portObj["index"],
                    "Attributes": gmshGroupIdList
                })

        return palaceTerminalObject

    def getGmshGroupId(self, groupName):
        return self._gmshGroupIdList[groupName]

    def getGmshGroupIdList(self):
        return self._gmshGroupIdList

    def addPort(self, name="", direction="-Z", R=50, excitation=False, type="lumped", index=1):
        self._portList[name] = {
            "name": name,
            "direction": direction,
            "R": R,
            "excitation": excitation,
            "type": type,
            "index": index,
            "objects": []
        }
        return

    def addObjectToPort(self, portName: str, objectName: str) -> None:
        if not "objects" in self._portList[portName].keys():
            self._portList[portName]["objects"] = []

        self._portList[portName]["objects"].append(objectName)
        self._portList[portName]["objects"] = list(set(self._portList[portName]["objects"]))

        return

    def addLumpedPart(self, name="", Rs:float=0.0, Ls:float=0.0, Cs:float=0.0):
        self._lumpedPartList[name] = {
            "name": name,
            "Rs": Rs,
            "Ls": Ls,
            "Cs": Cs,
            "objects": []
        }
        return

    def addObjectToLumpedPart(self, lumpedPartName: str, objectName: str) -> None:
        if not "objects" in self._lumpedPartList[lumpedPartName].keys():
            self._lumpedPartList[lumpedPartName]["objects"] = []

        self._lumpedPartList[lumpedPartName]["objects"].append(objectName)
        self._lumpedPartList[lumpedPartName]["objects"] = list(set(self._lumpedPartList[lumpedPartName]["objects"]))

        return

    def addConductivity(self, name="", conductivity:float=0.0, permeability:float=0.0, thickness:float=0.0):
        self._conductivityList[name] = {
            "name": name,
            "Conductivity": conductivity,
            "Permeability": permeability,
            "Thickness": thickness,
            "objects": []
        }
        return

    def addObjectToConductivity(self, conductivityName: str, objectName: str) -> None:
        if not "objects" in self._conductivityList[conductivityName].keys():
            self._conductivityList[conductivityName]["objects"] = []

        self._conductivityList[conductivityName]["objects"].append(objectName)
        self._conductivityList[conductivityName]["objects"] = list(set(self._conductivityList[conductivityName]["objects"]))

        return

    def gmshCreatePlate(self,
                     origin: tuple[float, float, float],
                     u: tuple[float, float, float],
                     v: tuple[float, float, float],
                     name: str | None = None):
            """A generalized 2D rectangular plate in XYZ-space.

            The plate is specified by an origin (o) in meters coordinate plus two vectors (u,v) in meters
            that span two of the sides such that all points of the plate are defined by:
                p1 = o
                p2 = o+u
                p3 = o+v
                p4 = o+u+v
            Args:
                origin (tuple[float, float, float]): The origin of the plate in meters
                u (tuple[float, float, float]): The u-axis of the plate
                v (tuple[float, float, float]): The v-axis of the plate
            """

            origin = np.array(origin)
            u = np.array(u)
            v = np.array(v)

            tagp1 = gmsh.model.occ.addPoint(*origin)
            tagp2 = gmsh.model.occ.addPoint(*(origin + u))
            tagp3 = gmsh.model.occ.addPoint(*(origin + v))
            tagp4 = gmsh.model.occ.addPoint(*(origin + u + v))

            tagl1 = gmsh.model.occ.addLine(tagp1, tagp2)
            tagl2 = gmsh.model.occ.addLine(tagp2, tagp4)
            tagl3 = gmsh.model.occ.addLine(tagp4, tagp3)
            tagl4 = gmsh.model.occ.addLine(tagp3, tagp1)

            tag_wire = gmsh.model.occ.addWire([tagl1, tagl2, tagl3, tagl4])

            tags: list[int] = [gmsh.model.occ.addPlaneSurface([tag_wire, ]), ]

            return tags

    ##THIS WORKS JUST FOR BOX!!! just to get box boundaries, not for sphere :( :( :( left here just for inspiration
    # def getBoundaryOuter(self, objectName, tol=1e-6) -> list[int]:
    #     internalGeometry = self.getGeometryObject(objectName)
    #     volumeTagList = [dimtag[1] for dimtag in internalGeometry["dimtags"] if dimtag[0] == 3]
    #
    #     outer = []
    #     for volumeTag in volumeTagList:
    #         # get bounding box
    #         xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
    #             3, volumeTag
    #         )
    #
    #         _, adjacent_surfaces = gmsh.model.getAdjacencies(3, volumeTag)
    #
    #         for surf_tag in adjacent_surfaces:
    #             sxmin, symin, szmin, sxmax, symax, szmax = \
    #                 gmsh.model.getBoundingBox(2, surf_tag)
    #
    #             # check if this surface lies on any face of the airbox bbox
    #             on_xmin = abs(sxmin - xmin) < tol and abs(sxmax - xmin) < tol
    #             on_xmax = abs(sxmin - xmax) < tol and abs(sxmax - xmax) < tol
    #             on_ymin = abs(symin - ymin) < tol and abs(symax - ymin) < tol
    #             on_ymax = abs(symin - ymax) < tol and abs(symax - ymax) < tol
    #             on_zmin = abs(szmin - zmin) < tol and abs(szmax - zmin) < tol
    #             on_zmax = abs(szmin - zmax) < tol and abs(szmax - zmax) < tol
    #
    #             if any([on_xmin, on_xmax, on_ymin, on_ymax, on_zmin, on_zmax]):
    #                 outer.append(surf_tag)
    #
    #     return outer

    #
    # I made this with assistance of Claude AI as I was kind tired and have no energy to think so deep at night :(
    #
    def getBoundaryOuter(self, objectName) -> list[int]:
        """
        Get outermost surface of object specified by name. It takes object volume tags from dimtags and takes their surfaces using gmsh
        .getAdjacencies(...) and looks how many boundary it touches, this should be working for boxes and sphere also.
        Args:
            objectName: str - specify object name in internal geometry manager

        Returns:
            surfaceTagList: list[int] - object surface tags which are object boundaries
        """
        internalGeometry = self.getGeometryObject(objectName)
        volumeTagList = [
            dimtag[1] for dimtag in internalGeometry["dimtags"]
            if dimtag[0] == 3
        ]
        volume_set = set(volumeTagList)

        seen = set()
        outer = []

        for volumeTag in volumeTagList:
            _, adjacent_surfaces = gmsh.model.getAdjacencies(3, volumeTag)

            for surf_tag in adjacent_surfaces:
                if surf_tag in seen:
                    continue
                seen.add(surf_tag)

                adj_vols, _ = gmsh.model.getAdjacencies(2, surf_tag)
                adj_vol_set = set(adj_vols.tolist())

                # surface is outer if every volume touching it
                # belongs to our object (no external neighbours)
                # and it is touched by exactly one volume
                external_neighbours = adj_vol_set - volume_set
                internal_neighbours = adj_vol_set & volume_set

                if len(external_neighbours) == 0 and len(internal_neighbours) == 1:
                    outer.append(surf_tag)

        return outer


    def validate_mesh_attributes(self):
        """Check for surfaces missing physical group assignment."""

        # get all surface tags in the mesh
        all_surfaces = set(
            tag for dim, tag in gmsh.model.getEntities(2)
        )

        # get all surfaces that are in a physical group
        tagged_surfaces = set()
        for dim, phys_tag in gmsh.model.getPhysicalGroups(dim=2):
            surfaces = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)
            tagged_surfaces.update(surfaces)

        # find untagged surfaces
        untagged = all_surfaces - tagged_surfaces

        if untagged:
            print(f"WARNING: {len(untagged)} surfaces have no physical group!")
            print(f"Untagged surface tags: {sorted(untagged)}")
            for s in untagged:
                bb = gmsh.model.getBoundingBox(2, s)
                print(f"  surface {s}: bbox {bb}")
        else:
            print("OK: all surfaces are in a physical group")

        # also check volumes
        all_volumes = set(tag for dim, tag in gmsh.model.getEntities(3))
        tagged_volumes = set()
        for dim, phys_tag in gmsh.model.getPhysicalGroups(dim=3):
            vols = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)
            tagged_volumes.update(vols)

        untagged_vols = all_volumes - tagged_volumes
        if untagged_vols:
            print(f"WARNING: {len(untagged_vols)} volumes have no physical group!")
            print(f"Untagged volume tags: {sorted(untagged_vols)}")
        else:
            print("OK: all volumes are in a physical group")

        return list(untagged), list(untagged_vols)

    def removeDuplicateTagsInGeometryObjects(self) -> None:
        """
        This goes from high priority objects to lower and removes common tags between them from lower priority, this method
        shouldn't exist if fragmentation method work right, but for now let's say it's easies sanitization to make model
        working.
        Returns:
        """

        self.sortGeomtriesBasedOnPriority()
        for k in range(len(self.geometryObjectList)-1):
            for m in range(k+1, len(self.geometryObjectList)):
                #remove all dimtags from higher priority object in lower priority object
                for dimtagToRemove in self.geometryObjectList[k]["dimtags"]:
                    #supress error when dimtag not exists in lower priority object
                    try:
                        self.geometryObjectList[m]["dimtags"].remove(dimtagToRemove)
                    except:
                        pass

        return

    def addFieldBall(self, VIn:float=5.0, VOut:float=15.0, Radius:float=60.0, XCenter:float=0.0, YCenter:float=0.0, ZCenter:float=0.0, Thickness:float=10.0) -> int:
        fieldId = gmsh.model.mesh.field.add("Ball")
        gmsh.model.mesh.field.setNumber(fieldId, "VIn", VIn)              # inside sphere
        gmsh.model.mesh.field.setNumber(fieldId, "VOut", VOut)            # outside sphere
        gmsh.model.mesh.field.setNumber(fieldId, "Radius", Radius)        # sphere radius
        gmsh.model.mesh.field.setNumber(fieldId, "XCenter", XCenter)
        gmsh.model.mesh.field.setNumber(fieldId, "YCenter", YCenter)
        gmsh.model.mesh.field.setNumber(fieldId, "ZCenter", ZCenter)
        gmsh.model.mesh.field.setNumber(fieldId, "Thickness", Thickness)  # transition zone
        return fieldId

    def removeDimtagsNotInModel(self, dimtagList:list[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Takes dimtag list and remove dimtags which are not available at model.
        Args:
            dimtagList:

        Returns:list[tuple[int, int]]: Dimtag list with dimtags available in model.
        """

        modelAllDimtags = gmsh.model.getEntities()
        return [dimtag for dimtag in dimtagList if dimtag in modelAllDimtags]

    def getGeometryObjectDimension(self, name: str) -> int:
        """
        Return wheteher object is 0D,1D,2D,3D. For now it iterates over object dimtags and checks their dimensionality, but
        geometry objects inside geometry manager have also 'type' parameter which could be 'stepfile' or 'volume' or something
        else.
        Args:
            name:str: Internal object name in geometry maanger

        Returns:objectDimension:int
        """

        geometryObject = self.getGeometryObject(name)
        dimtagList = self.removeDimtagsNotInModel(geometryObject["dimtags"])

        objectDimension = -1

        # First check if object is probably 2D, 3D
        for dimtag in dimtagList:
            if dimtag[0] == 2:
                objectDimension = 2
            if dimtag[0] == 3:
                objectDimension = 3

        # when there are no 2D or 3D tags then check if object is 0D, 1D diemnsion means it's curve or points
        if objectDimension == -1:
            for dimtag in dimtagList:
                if dimtag[0] == 0:
                    objectDimension = 0
                if dimtag[0] == 1:
                    objectDimension = 1

        return objectDimension

    def getGeometryObjectEdges(self, name: str) -> list[tuple[int, int]]:
        geometryObject = self.getGeometryObject(name)
        dimtagList = self.removeDimtagsNotInModel(geometryObject["dimtags"])
        objectDimension = self.getGeometryObjectDimension(name)

        objectEdgeDimtags = []
        if objectDimension == 3:
            objectSurfaceDimtags = gmsh.model.getBoundary(dimtagList, combined=False, oriented=False, recursive=False)
            objectEdgeDimtags = gmsh.model.getBoundary(objectSurfaceDimtags, combined=False, oriented=False, recursive=False)
        elif objectDimension == 2:
            objectEdgeDimtags = gmsh.model.getBoundary(dimtagList, combined=False, oriented=False, recursive=False)
        elif objectDimension == 1:
            objectEdgeDimtags = [dimtag for dimtag in geometryObject["dimtags"] if dimtag[0] == 1]

        return objectEdgeDimtags
