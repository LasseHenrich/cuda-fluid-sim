# Simulation theory
Knowledge drawn mostly from [this **GPU Gems** guide](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu) and [this visualization](https://jamie-wong.com/2016/08/05/webgl-fluid-simulation/).

## Representation
Most important is the velocity of a fluid for simulating it, so we represent the velocity in a vector field, i.e. a 2(or 3)-dimensional grid &rarr; for every position $\vec x=(x,y)$, there is an associated velocity at time $t$ &rarr; $\vec u(\vec x,t)=(u(\vec x,t),v(\vec x,t))$. In every time step we update the grid with the correct velocity values by solving the Navier-Stokes equations.

## Navier-Stokes equations
The fluid is represented by its velocity field $\vec u(\vec x,t)$ and a scalar pressure field $p(\vec x,t)$, which vary in space and time &rarr; If we know velocity and pressure for $t=0$ and assume incompressibility and homogeneity, the fluid's state over time can be described by the Navier-Stokes equations for incompressible flow. We can break them up into the following four terms:

1. **Advection**
Transfer of a property from one place to another due to the motion of the fluid. That means, if you put dye into moving water, the dye will be pulled (*advected*) through it. Note that the velocity of a fluid carries *itself* along as well, which is called *self-advection*.
1. **Pressure**
Builds up since particles move around and push each other. Any pressure naturally leads to acceleration.
1. **Diffusion**
Viscosity is a measure of how *thick* and therefore *resistive* to flow a fluid is. This resistance results in diffusion of the momentum.<br>
&rarr; We can drop this for now by pretending that our fluid's viscosity is zero.
1. **External Forces** 
Possibly various external forces like a fan blowing air.

Note that *incompressible* just means that the volume of any subregion of the fluid is constant over time. It is *homogeneous* if its density is constant in space. Both assumptions together mean that density is constant in time and space.

### Solving the Navier-Stockes equations
**Divergence** is the rate at which *density* exits a given region of space. Here, it measures the net change of velocity across a surface. To enforce the incompressibility assumption, the fluid always has to have zero divergence, hence we need a divergence-*free* vector field.

The NV equations are three equations that we can solve for the quantities $u,v,p$. We can transform them using the *Helmholtz-Hodge Decomposition Theorem*, which states that any vector field can be decomposed into the sum of (i) a divergence-free vector field, and (ii) the gradient of a scalar field. Ref. the GPU Gems guide for more details, but we basically end up with

$$
\frac{\partial \vec u}{\partial t} = \mathbb P(-(\vec u\cdot\nabla)\vec u+\nu\nabla^2\vec u+ F),
$$

where $\mathbb P$ is a projection operator that projects a vector field $\vec w$ into its divergence-free component $\vec u$, $\nu$ is the viscosity, and $F$ are the external forces. So, from left to right inside the parentheses, we compute the advection, diffusion, and force terms separately. However, in a typical implementation, the solution is found by compositing operators for advection ($\mathbb A$), diffusion ($\mathbb D$), force application ($\mathbb F$), and projection ($\mathbb P$), where each operator takes a field as an input and produces a field as an output after a step, so a simulation step is

$$
    \mathbb S(u) = \mathbb P\circ\mathbb F\circ\mathbb D\circ\mathbb A(u),
$$

with $t$ omitted for clarity.

## Discrete steps
For each step, we do
```
u = advect(u)
u = diffuse(u)
u = addForces(u)
p = computePressure(u)
u = subtractPressureGradient(u,p)
```

### Advection
(Ref. GPU Gems 38.3.3)
Rather than advecting quantities by computing where a particle moves over the current time step, we trace the trajectory of the particle from each grid cell back in time to its former position and copy the quantities at that position to the starting grid cell. To update a quantity $q$, which could be velocity, density, temperature etc., we use

$$
    q(\vec x, t+\Delta t)=q(\vec x-\vec u(x,t)\Delta t, t).
$$

### Diffusion

$$
(\textbf I-\nu\Delta t\nabla^2)\vec u(\vec x,t+\Delta t) = \vec u(\vec x,t),
$$

where $\textbf I$ is the identity matrix.

### Pressure
We need to find the pressure $p$ such that its Laplacian is the divergence: $\nabla^2p = \text{div}$, and we also know that $\nabla^2p$ at a cell must be equal to the sum of the four neighboring cells' pressures minus 4 times the original cells pressure. We can use the Jacobi iteration to solve this by repeatedly making each cell locally consistent:

$$
x_{i,j}^{(k+1)}=\frac{x_{i-1,j}^{(k)}+x_{i+1,j}^{(k)}+x_{i,j-1}^{(k)}+x_{i,j+1}^{(k)}+\text{div}}{4}.
$$

## Boundary Conditions
We need to determine how to compute values at the edges of the simulation domain: For now, we assume that the fluid is in a box and cannot flow through its sides &rarr; velocity goes to zero at the boundaries (*no-slip* condition), and the rate of change of pressure in the direction normal to the boundary is zero as well (*pure Neumann*);

# Implementation

## Grid Representation
Standard would be using a **collocated** grid, where velocity components and pressure live at the exact same pixel centers. However, with this, spatial derivatives for cell $i$ will look at indices $i+1$ and $i-1$, skipping the cell itself, which may result in **checkerboard instability**.

&rarr; Instead, we may need to use a **Marker and Cell** (MAC) grid layout, i.e. storing the pressure at the cell centers, but shifting horizontal velocity components to the vertical cell faces and vertical velocity components to the horizontal cell faces. Note that this matches the *Staggered Grid* described in the GPU Gems guide 38.5.3, reducing numerical oscillations and increasing the accuracy of many calculations.

> **Not implemented!** A collocated grid seems to work fine for now

## Double buffering (ping-pong)
A stencil can't safely read and write to the same field, since this could lead to race conditions. So, we use double buffering with a pointer swap, i.e. reading from buffer A, writing to buffer B, swapping the pointers for the next pass. &rarr; We have two buffers for velocity, pressure and dye. Convention is index 0 = current state for reading, index 1 = next state for writing.
(In WebGL this is called [render to texture](https://webglfundamentals.org/webgl/lessons/webgl-image-processing-continued.html))

## Kernels
(Ref. GPU Gems 38.3.3)

### Advection
As described in the theory section and shown by the equation, instead of pushing dye forward out of each cell, each cell checks where its fluid came from and pulls the dye from there. This means that every thread *writes exactly one cell* and *only reads others* &rarr; this is perfect for a GPU implementation. In contrast, in a  *forward* setting, multiple threads may target the same cell, which would result in race conditions be need to be solved via atomics.

Note that since the *source* position mostly lands between cell centers, we bilinearly interpolate the 4 surrounding cells. Also, note that this numerical error causes some diffusion, which is wanted and not a bug (ref. GPU Gems 38.4.1; they don't actually implement separate diffusion).

### Projection
The projection step is divided into two operations, solving the Poisson-pressure equation, then subtracting the gradient of the pressure from the velocity field. We therefore need kernels for computing the divergence of the velocity field, the Jacobi iteration program, and for subtracting the pressure gradient from the velocity field:
1. The divergence kernel computes $\text{div} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$ with finite differences:
`div(x,y) = 0.5 * (vel[x+1].x − vel[x−1].x) + 0.5 * (vel[y+1].y − vel[y−1].y)`<br>
1. As described in the theory part, the pressure equation is solved with the Jacobi iteration. Applying it 40-80 times yields a good result, acc. to GPU Gems; we observed that 40 is enough.
1. Then, a gradient subtraction kernel subtracts the pressure's gradient from the velocity. The gradient is again computed using finite differences.
1. Lastly, we have a tiny kernel which enforces the no-slip boundary condition described earlier, setting the velocity to zero on the boundary.

(ref. GPU Gems guide table 38-1 for finite difference formulas and 38.3.3 §Projection for more explanations)

All three projection kernels perform **output-centric** decompositions via **stencil computation** (look up *Iterative Stencil Loop* (ISL)). That means, that each thread is computing the output value of one grid element by performing a computation over the neighborhood of that element.

#### Simple Pressure evaluation (Jacobi Iteration)
As described, we calculate the pressure by applying a Jacobi iteration 40 times, each iteration being executed via a separate kernel call and the pointers to the pressure buffer being swapped in between two calls. Note that this double-buffering is very efficient and $\mathcal{O}(1)$, since we're just swapping host pointers that point to device arrays.

It would be tempting to avoid the 40 different kernel launches by moving the `for`-loop to the kernel, i.e. something like:

```cpp
__global__ void fullJacobiIteration(int iterationCount, ...) {
    for (int i = 0; i < iterationCount; i++)  {
        singleJacobiIteration();
        __syncthreads();
        swapBuffers();
    }
}
```
This would be incorrect, since `__syncthreads()` only synchronizes threads in the same block. However, the stencil computation requires that *every* block has completed iteration `i` before moving on, because cells at a block boundary depend on outputs being produced by neighboring blocks. Instead, launching 40 kernels for the 40 iteration within a single stream guarantees that iteration `i` has finished before `i+1` starts.

Also, there is practically no overhead for the 40 kernel launches. And again, please note that the `pressure[0]` and `pressure[1]` buffers live on the GPU at all times and do not have to be copied between host and device during launches.

#### Optimization for Pressure evaluation
##### 1. Simple Tiling
As one performance optimization, we use *tiling*, i.e. loading a chunk of multiple neighborhoods (a *tile*) into shared memory, which improves data locality and reduces memory access overhead. Threads collaboratively load a tile (plus its *halo* region) into shared memory, perform the computation locally, and then write the results back to global memory. One thread block is assigned to compute exactly one tile.

Note that this alone actually *decreased* performance (see (measurements/Measurements.md)[Measurements.md]), since thread utilization goes down dramatically: In 3D, with a $8^3$ sized block, interior threads are only $(6/8)^3=42\%$. Slightly better at $10^3$ sized blocks with $51\%$, but still bad. Furthermore, with tiles of size $8^3$ and a $128^3$ grid, suddenly $\left\lceil\frac{128}{6}\right\rceil^3=10648$ blocks have to be launched, versus $\left(\frac{128}{8}\right)^3=4096$ without tiling, resulting in significant scheduling overhead.

##### 2. Slab/2.5D
Standard tiling has bad properties in 3D, like a small fraction of threads producing outputs, as well as a large number of threads and blocks having to be launched.

The *Slab* method (aka. *2.5D* method) addresses both these problems: Each thread block is now responsible for one 2D tile, which it moves and calculates through the z-dimension of the grid. Therefore, we need less blocks (like in a $N^2$ instead of a $N^3$), and each thread does $N$ times the work &rarr; **Coarsening**. Because of the dimensionality reduction, it also increases the fraction of output threads: With $10^2$ sized tiles, the output fraction is $(8/10)^2=64\%$.

In 2D, we can now also increase the block size per dimension, which also makes us think about **coalescing**: We can increase one dimension to 32, mapping perfectly into one warp. Setting the other dimension to 32 as well is possible and would provide perfect halo utilization, but I observed slightly better performance with 32x8, probably because 32x32 uses the maximum of 1024 threads per block, limiting scheduling flexibility. Similarly, 16x16, which exhibits a high output thread fraction but warp divergence, is noticeably slower. The output thread utilization for 32x8 is still at a good $180/256=70\%$.

##### 3. Red-Black Gauss-Seidel
Slab/2.5D still doesn't address the biggest performance overhead, which is repeated reads and writes of the full field in device global memory across 40 iterations. *Red-Black Gauss-Seidel* (RBGS) can help:

Here, the idea is to split the grid into a 3D checkerboard of *red* and *black* cells. Per iteration, we first update all red cells using the current black neighbors, and then vice versa. Updates can be done in-place, since red cells only read black cells and black ones only red ones. Note that this is not really a *Jacobi* iteration.

A Jacobi iteration updates all cells simultaneously with old neighbor values. The argument for RBGS is that updated values are used *immediately*, so the convergence rate should double, i.e. the necessary iteration count (and memory accesses) halves for roughly the same accuracy. However, implementing this naively (as in `rbgsKernel_simple`) will result in a big *memory access coalescing* issue: Because of the checkerboarding, the warp of 32 threads will need 2x the memory it should need!

So, a better solution (ref. `rbgsKernel_coalesced`) packs all red and all black cells into their own contiguous arrays, ensuring that consecutive threads access contiguous memory addresses, eliminating divergence

#### A note on profiling
The Nvidia Nsight Compute tool doesn't support Pascal GPUs anymore, which my GTX1060 belongs to. I also didn't get it to work with the 2019.5.1 version, which should support Pascal.

## Main Loop
The main loop functions like a *game loop*, i.e. polls click events, renders, then prepares the next frame.

### Objects for CUDA&rarr;OpenGL communication
We're trying to build a bridge between OpenGL and CUDA using the following objects.

#### `glTexture`
512x512 (grid size) allocation of raw VRAM that hold pixel color data (i.e. 4 bytes per pixel) &rarr; basically the canvas.

#### `glTextureCudaHandle`
**registration bridge** that allows CUDA to temporarily lock `glTexture` and rewrite its pixels.

#### `blitFBO`
Helps with glitting, i.e. it wraps around the texture so the hardware can copy (*blit*) it directly to the monitor.

### Timing
In order to compare performance of pre- with post-optimization per kernel. We need to time CUDA kernels independently from each other without stalling the CPU Code. For this, we have a struct `CudaTimer` which uses events as shown in [this Cuda guide](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/).



# Libraries

## GLFW
Cross-platform utility that talks to OS for handling window creation, context management, keyboard/mouse inputs.

## GLEW
OpenGL Extension Wrangler. Dynamically queries physical Nvidia driver at runtime and hooks up all modern OpenGL function pointers.