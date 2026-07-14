# Work Log
## Preparation
1. **Tech stack:** Researching how to write a GUI with an efficient bridge to CUDA and setting up the project with cmake.
1. **Working example showcasing CUDA/OpenGL working together:** With the plasma kernel specifically created to set up and test the pipeline from CUDA to OpenGL.
1. **Simulation Theory:** Reading through multiple resources online and used AI some understand the basics of fluid simulation, disregarding any (GPU) implementation details. A significant portion of the [docs](Docs.md) was written here.

## First, simple 2D implementation
1. **Field buffers + a field-to-color kernel:** Defining the central FluidFields struct, writing a function for allocating and freeing, replacing the plasma kernel with a kernel that maps dye to colors on the texture
1. **Dye injection:** Injecting a Gaussian blob of dye at the cursor while clicked &rarr; early interactivity for testing.
1. **Dye Advection kernel:** A simple (unoptimized) advection kernel, tested with a vortex velocity field.
1. **Velocity self-advection + mouse force injection:** Another, similar advection kernel, now for the velocity, and added functionality to inject force with the mouse (alongside the already existing dye injection).
1. **Projection:** Writing kernels for calculating divergence, pressure, and subtracting the pressure gradient from the velocity.

## Extension to 3D, still simple
1. **Timing:** IN 2D, the kernels together need only 1-3 milliseconds on a 512x512 grid. I expect this to increase a lot when going to 3D, so have a comparison (and also for later optimization in 3D), the `CudaTimer` is introduced.
1. **Refactoring**: separate helpers for bi/trilinear interpolation, backtracing, kernel launch configuration, and more.
1. **Moving to 3D**: FluidFields (allocation), kernel launch configurations, trilinear interpolation for advection, 6 neighbors for projection computation, rendering the center slice, mouse interaction in the slice within 3D.
1. **3D visualization:** OpenGL doesn't provide a simple *draw this volume* call. So instead, a simple (unoptimized) ray marching algorithm for visualizing the liquid, with some common light calculation, was implemented.

## Simple GUI
1. **ImGUI integration**: Installation + integration and tested with a first dropdown to select the render mode.
2. **GUI Controls**: Exposing sliceZ, force injection with strength, dye injection with radius+strength.

## Optimizations
1. **Tiling**: Simple tiling approach as shown in the lecture and exercise 3. Shown to not be materially better than simple/naive approach.
1. **Next steps evaluation**, what other/further optimization might bring more benefit than simple tiling.
1. **GUI Dropdown** for selecting the jacobi iteration algorithm, so far either simple and tiling, later slab and rbgs.
1. **Red-Black Gauss-Seidel**: Faster technique for evaluating the pressure, implemented in both a simple and a well coalesced way.
1. **Slab method**: Advanced tiling method for increasing the fraction of output threads and decreasing the number of launched threads and blocks compared to standard tiling.