passer
======

Converts GPU torch7 models to CPU equivalent models.
The current version is very limited in what it is able to convert, although the code is easy to extend.

Installation
============
To install via luarocks, run:

    luarocks install --server=http://luarocks.org/manifests/williford passer

Or clone the repository:

    git clone git@github.com:williford/passer.torch.git

and then copy the passer.lua file to your working folder.

In order to use:

    local passer = require 'passer'


Usage
=====

The conversion needs to be done on a system that has a GPU (that is able to load the required model).

```lua
require 'nn';
require 'cudnn';

-- do not put "local" if in interactive session!
local passer = require 'passer'

model = torch.load('/path/to/model_xx.t7')
cpu_model = passer.tocpu(model)
torch.save('cpu_model_xx.t7')
```
