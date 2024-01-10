# pyecsca-notebook

![License: MIT](https://img.shields.io/github/license/J08nY/pyecsca-notebook.svg)

**Py**thon **E**lliptic **C**urve cryptography **S**ide-**C**hannel **A**nalysis toolkit.

Notebook package, see below for description of the notebooks showcasing the toolkit.
See the [main repo](https://github.com/J08nY/pyecsca) for more information.

## Notebooks

### Configuration space

The [configuration space](configuration_space.ipynb) notebook explores the size of the space of
possible implementation configurations of ECC.

### Simulation

The [simulation](simulation.ipynb) notebook showcases the simulation and execution tracing capabilities
of the toolkit.

### Codegen & emulation

The [codegen](codegen.ipynb) notebook demonstrates the process of generating and interacting with
generated C implementations of ECC for micro-controllers. The generated implementations can either
be run on compatible hardware or emulated (at CPU-level) using the
[Rainbow](https://github.com/Ledger-Donjon/rainbow)-based emulator demonstrated in the
[emulator](emulator.ipynb) notebook.

### Measurement

The [measurement](measurement.ipynb) notebook demonstrates the trace acquisition using
PicoScope/ChipWhisperer scopes that can be used with the toolkit.

### Visualization

The [visualization](visualization.ipynb) notebook showcases the trace visualization capabilities
of the toolkit.

### Smartcards

The [smartcards](smartcards.ipynb) notebook shows the options of communicating with smartcard
targets using the toolkit.

### Reverse-engineering

#### RPA-RE

The [RPA](re/rpa.ipynb) notebook uses the Refined Power Analysis attack-based technique to reverse-engineer
the scalar multiplier of ECC implementations, given access to a power side-channel.

#### EPA-RE

The [EPA](re/epa.ipynb) notebook uses the ideas behind the Exceptional Procedure Attack to reverse-engineer
the coordinate system and formulas of ECC implementations, given access to an error side-channel.

#### Structural

The [structural](re/structural.ipynb) notebook explores the structure of scalar multiplers and addition
formulas for reverse-engineering purposes.

## License

    MIT License

    Copyright (c) 2018-2023 Jan Jancar
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    

*Development is supported by the Masaryk University grant [MUNI/C/1701/2018](https://www.muni.cz/en/research/projects/46834),
this support is very appreciated.*
