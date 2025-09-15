API Reference
=============

.. doxygenclass:: scve::VoxelEngine
   :project: simple-cuda-voxel-engine
   :members:

.. doxygenclass:: scve::Vector3
   :project: simple-cuda-voxel-engine
   :members:

.. literalinclude:: ../include/scve/internal/structure/functor.h
   :project: simple-cuda-voxel-engine
   :language: cuda
   :lines: 9-13

.. doxygenclass:: scve::XYZFrameToIdFunctor
   :project: simple-cuda-voxel-engine
   :members:

The functor you can pass to insertVoxels, provides a position + frame number to block id mapping

.. doxygenclass:: scve::IdFrameToMaterialFunctor
   :project: simple-cuda-voxel-engine
   :members:

.. literalinclude:: ../include/scve/internal/structure/functor.h
   :project: simple-cuda-voxel-engine
   :language: cuda
   :lines: 15-19

The functor you can pass to setMaterials, provides a block id + frame number to material mapping

.. doxygenclass:: scve::Material
   :project: simple-cuda-voxel-engine
   :members:

.. literalinclude:: ../include/scve/internal/structure/material.h
   :project: simple-cuda-voxel-engine
   :language: cuda
   :lines: 7-20

The Material used by the Phong illumination model. You can set the **color**, **diffuse** which means the diffuse reflection that makes objects
look more opaque, **specular** that makes them appear shiny and **specularExponent** that controls the size of the light spots.