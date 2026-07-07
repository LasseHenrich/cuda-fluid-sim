# Work Log
## Preparation
1. **Tech stack:** Researching how to write a GUI with an efficient bridge to CUDA and setting up the project with cmake.
1. **Working example showcasing CUDA/OpenGL working together:** With the plasma kernel specifically created to set up and test the pipeline from CUDA to OpenGL.
1. **Simulation Theory:** Reading through multiple resources online and used AI some understand the basics of fluid simulation, disregarding any (GPU) implementation details. A significant portion of the [docs](Docs.md) was written here.

## First, simple implementation
1. **Field buffers + a field-to-color kernel:** Defining the central FluidFields struct, writing a function for allocating and freeing, replacing the plasma kernel with a kernel that maps dye to colors on the texture
1. **Dye injection:** Injecting a Gaussian blob of dye at the cursor while clicked &rarr; early interactivity for testing.
1. **Dye Advection kernel:** A simple (unoptimized) advection kernel, tested with a vortex velocity field.
1. **Velocity self-advection + mouse force injection:** Another, similar advection kernel, now for the velocity, and added functionality to inject force with the mouse (alongside the already existing dye injection).
1. **Projection:** Writing kernels for calculating divergence, pressure, and subtracting the pressure gradient from the velocity.

## Extension to 3D
1. **Timing:** IN 2D, the kernels together need only 1-3 milliseconds on a 512x512 grid. I expect this to increase a lot when going to 3D, so have a comparison (and also for later optimization in 3D), the `CudaTimer` is introduced.

## Simple GUI
We should have some GUI to play around with the simulation early on, since this will help in the debugging/optimization stage immensely.