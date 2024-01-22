# Focused ultrasound simulation

[![HIFU CI](https://github.com/adeebkor/fenicsx-fus/actions/workflows/python-app.yml/badge.svg)](https://github.com/adeebkor/fenicsx-fus/actions/workflows/python-app.yml)

FEniCSx implementation for focused ultrasound (FUS) simulation. 

## Introduction

A focused ultrasound (FUS) technique is a non-invasive medical 
technique that uses ultrasound for thermal ablation.
FUS generates acoustic waves using well-placed transducers to heat 
unwanted tissue while keeping the surrounding tissue unaffected.

## Solver method

* Spatial discretisation:
    * High-order finite element method.
* Time discretisation:
    * Classical Runge-Kutta 4th order method.

## Solver design

* Requires full hexahedral (3D) or quadrilateral (2D) mesh.
* Implementation of the sum-factorisation algorithm for
tensor-product element.
