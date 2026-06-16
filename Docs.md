# Main Loop
We're trying to build a bridge between OpenGL and CUDA using the following objects.

## Objects

### `glTexture`
800x600 (window size) allocation of raw VRAM that hold pixel color data (i.e. 4 bytes per pixel) &rarr; basically the canvas.

### `glTextureCudaHandle`
**registration bridge** that allows CUDA to temporarily lock `glTexture` and rewrite its pixels.

### `blitFBO`
Helps with glitting, i.e. it wraps around the texture so the hardware can copy (*blit*) it directly to the monitor.



# Libraries

## GLFW
Cross-platform utility that talks to OS for handling window creation, context management, keyboard/mouse inputs.

## GLEW
OpenGL Extension Wrangler. Dynamically queries physical Nvidia driver at runtime and hooks up all modern OpenGL function pointers.