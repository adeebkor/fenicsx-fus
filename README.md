# High-intensity focused ultrasound simulation

[![HIFU CI](https://github.com/adeebkor/hifu-simulation/actions/workflows/python-app.yml/badge.svg)](https://github.com/adeebkor/hifu-simulation/actions/workflows/python-app.yml)

FEniCSx implementation for high-intensity focused ultrasound (HIFU) simulation. 

## Introduction

A high-intensity focused ultrasound (HIFU) technique is a non-invasive medical 
technique that uses ultrasound for tissue heating and ablation.
HIFU generates acoustic waves using well-placed transducers to heat 
unwanted tissue while keeping the surrounding tissue unaffacted.

## HIFU setup

Consider a HIFU simulation domain of 20cm by 20cm by 20cm. This setting is
typical for a simulation of abdominal region such as simulation on the kidney 
or the liver. A HIFU transducer usually operates at 0.25MHz to 5MHz. Using the
speed of sound in water to be 1480m/s and a transducer that operates at 1MHz,
we obtain the fundamental wavelength to be 1.48mm. To accurately simulate this
setting with a high order finite element method, a suitable grid resolution
is required to fully resolve the wave pattern. For example, for a p4 element, 
through numerical experimentation, the number of element per wavelength
required is 4. In this case, for a simulation domain of 20cm by 20cm by 20cm
with fundamental wavelength of 1.48mm, the number of elements required is
approximately $160 \times 10^{6}$. This translates to approximately $10^{10}$
degrees of freedom.

Advantages of our approach:
* The handling of complex shape and region is trivial, e.g. different density
region such as fluids and liver.
* Utilisation of high degree basis reduces dispersion error.
* Flexibility of using different source fields. We are not restricted to
periodic source field.

# Performance estimate

An accurate representation of HIFU operating condition is typically model 
using the Westervelt equation with the associated boundary and initial 
conditions. A variational formulation of these set of equations generate
several computational kernels as follow

![alt text](westervelt_forms.png)
