-- Modified from Michael Partheil
-- (https://groups.google.com/forum/#!topic/torch7/i8sJYlgQPeA)
local passer = {
  _VERSION = 'passer.torch 0.0.0',
  _URL     = 'https://github.com/williford/passer.torch',
  _DESCRIPTION = 'Converts GPU torch models to CPU equivalent versions',
  _LICENSE = 'MIT'
}

function passer.replace_modules(net, orig_class_name, replacer)
  local nodes, container_nodes = net:findModules(orig_class_name)
  for i = 1, #nodes do
    for j = 1, #(container_nodes[i].modules) do
      if container_nodes[i].modules[j] == nodes[i] then
        local orig_mod = container_nodes[i].modules[j]
        container_nodes[i].modules[j] = replacer(orig_mod)
      end
    end
  end
end

function passer.tocpumodel(net)
  local net_cpu = net:clone():float()

  replace_modules(net_cpu, 'cudnn.SpatialConvolution', 
    function(orig_mod)
      local cpu_mod = nn.SpatialConvolutionMM(orig_mod.nInputPlane, orig_mod.nOutputPlane,
          orig_mod.kW, orig_mod.kH, orig_mod.dW, orig_mod.dH, orig_mod.padW, orig_mod.padH)
      cpu_mod.weight:copy(orig_mod.weight)
      cpu_mod.bias:copy(orig_mod.bias)
      return cpu_mod
    end)
  replace_modules(net_cpu, 'cudnn.SpatialMaxPooling',
    function(orig_mod)
      local cpu_mod = nn.SpatialMaxPooling(
        orig_mod.kW,
        orig_mod.kH,
        orig_mod.dW,
        orig_mod.dH,
        orig_mod.padW,
        orig_mod.padH)
      return cpu_mod
    end)

  replace_modules(net_cpu, 'cudnn.ReLU', function() return nn.ReLU() end)

  return net_cpu
end

return passer
