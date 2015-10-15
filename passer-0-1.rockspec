package = "passer"
version = "0-1"

source = {
  url = "git://github.com/williford/passer.torch.git"
}

description = {
  summary = "Converts GPU torch models to CPU equivalent versions",
  detailed =[[
Converts GPU torch models to their CPU equivalents, for those unfortunate souls that don't have GPUs on all their development computers.

Based on code by Michael Partheil (https://groups.google.com/forum/#!topic/torch7/i8sJYlgQPeA).
  ]],
  homepage = "https://github.com/williford/passer.torch",
  license = "MIT"
}

dependencies = {
  "torch",
  "nn"
}

build = {
  type = "builtin",
  modules = {
    passer = "passer.lua"
  }
}
