# Focused ultrasound simulation

[![FEniCSx-FUS CI](https://github.com/adeebkor/fenicsx-fus/actions/workflows/python-app.yml/badge.svg)](https://github.com/adeebkor/fenicsx-fus/actions/workflows/python-app.yml)

FEniCSx implementation for focused ultrasound (FUS) simulation. 

## Introduction

Focused ultrasound (FUS) is a non-invasive medical technique that uses
ultrasound for therapeutic purposes. FUS generates acoustic waves using
transducers to treat diseases inside a patient body without incisions.

## Solver method

* Spatial discretisation:
    * High-order finite element method.
* Time discretisation:
    * Classical Runge-Kutta 4th order method.

## Solver design

* Requires full hexahedral (3D) or quadrilateral (2D) mesh.
* Implementation of the sum-factorisation algorithm for
tensor-product element.
