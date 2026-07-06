# Simulation theory
Mostly from [this **GPU Gems** guide](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu) and [this visualization](https://jamie-wong.com/2016/08/05/webgl-fluid-simulation/).

## Representation
Most important is the velocity of a fluid for simulating it, so we represent the velocity in a vector field, i.e. a 2(or 3)-dimensional grid &rarr; for every position $\vec x=(x,y)$, there is an associated velocity at time $t$ &rarr; $\vec u(\vec x,t)=(u(\vec x,t),v(\vec x,t),w(\vec x,t))$. In every time step we update the grid with the correct velocity values by solving the Navier-Stokes equations.

## Navier-Stokes equations
The fluid is represented by its velocity field $\vec u(\vec x,t)$ and a scalar pressure field $p(\vec x,t)$, which vary in space and time &rarr; If we know velocity and pressure for $t=0$, the fluid's state over time can be described by the Navier-Stokes equations. We can break them up into the following four terms:

1. **Advection**
Transfer of a property from one place to another due to the motion of the fluid. I.e., if you put dye into moving water, the dye will be pulled (*advected*) through it.
1. **Pressure**
Builds up since particles move around and push each other. Any pressure naturally leads to acceleration.
1. **Diffusion**
Viscosity is a measure of how *thick* and therefore *resistive* to flow a fluid is. This resistance results in diffusion of the momentum.<br>
&rarr; We can drop this for now by pretending that our fluid's viscosity is zero.
1. **External Forces** 
Possibly various external forces like a fan blowing air.<br>
&rarr; We can also ignore this for now.

### Solving the Navier-Stockes equations
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
$$
x_{i,j}^{(k+1)}=\frac{x_{i-1,j}^{(k)}+x_{i+1,j}^{(k)}+x_{i,j-1}^{(k)}+x_{i,j+1}^{(k)}+\alpha b_{i,j}}{\beta},
$$
where $\alpha,\beta$ are constants.

## Boundary Conditions
We need to determine how to compute values at the edges of the simulation domain: For now, we assume that the fluid is in a box and cannot flow through its sides &rarr; velocity goes to zero at the boundaries (*no-slip* condition), and the rate of change of pressure in the direction normal to the boundary is zero as well.

# Implementation

## Grid Representation
Standard would be using a **collocated** grid, where velocity components and pressure live at the exact same pixel centers. However, with this, spatial derivatives for cell $i$ will look at indices $i+1$ and $i-1$, skipping the cell itself, which may result in **checkerboard instability**.

&rarr; Instead, we may need to use a **Marker and Cell** (MAC) grid layout, i.e. storing the pressure at the cell centers, but shifting horizontal velocity components to the vertical cell faces and vertical velocity components to the horizontal cell faces. Note that this matches the *Staggered Grid* described in the GPU Gems guide 38.5.3, reducing numerical oscillations and increasing the accuracy of many calculations.

> First, I'll implement the collocated grid, possibly upgrading to MAC later

## Double buffering (ping-pong)
A stencil can't safely read and write to the same field, since this could lead to race conditions. So, we use double buffering with a pointer swap, i.e. reading from buffer A, writing to buffer B, swapping the pointers for the next pass. &rarr; We have two buffers for velocity, pressure and dye. Convention is index 0 = current state for reading, index 1 = next state for writing.
(In WebGL this is called [render to texture](https://webglfundamentals.org/webgl/lessons/webgl-image-processing-continued.html))

## Main Loop Details

### Objects
We're trying to build a bridge between OpenGL and CUDA using the following objects.

#### `glTexture`
512x512 (grid size) allocation of raw VRAM that hold pixel color data (i.e. 4 bytes per pixel) &rarr; basically the canvas.

#### `glTextureCudaHandle`
**registration bridge** that allows CUDA to temporarily lock `glTexture` and rewrite its pixels.

#### `blitFBO`
Helps with glitting, i.e. it wraps around the texture so the hardware can copy (*blit*) it directly to the monitor.



# Libraries

## GLFW
Cross-platform utility that talks to OS for handling window creation, context management, keyboard/mouse inputs.

## GLEW
OpenGL Extension Wrangler. Dynamically queries physical Nvidia driver at runtime and hooks up all modern OpenGL function pointers.