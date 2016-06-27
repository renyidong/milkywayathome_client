arg = { ... }

assert(#arg == 4, "Expected 4 arguments")
assert(argSeed ~= nil, "Expected seed")

prng = DSFMT.create(argSeed)

evolveTime       = arg[1]
reverseOrbitTime = arg[1] / arg[2]

r0  = arg[3]

dwarfMass  = arg[4]

model1Bodies = 2
totalBodies = model1Bodies

nbodyLikelihoodMethod = "EMD"
nbodyMinVersion = "1.32"

function makePotential()
   return  nil
end

encMass = plummerTimestepIntegral(r0, sqr(r0) + sqr(r0/arg[4]) , dwarfMass, 1e-7)

function makeContext()
   return NBodyCtx.create{
      timeEvolve = evolveTime * sqr(1/10.0) * sqrt((pi_4_3 * cube(r0)) / (encMass + dwarfMass)),
      timestep   = sqr(1/10.0) * sqrt((pi_4_3 * cube(r0)) / (encMass + dwarfMass)),
      eps2       = calculateEps2(totalBodies, r0),
      criterion  = "Exact",
      useQuad    = true,
      theta      = 1.0
   }
end

-- Also required
function makeBodies(ctx, potential)
    local firstModel
    local finalPosition, finalVelocity = Vector.create(0,0,0), Vector.create(0,0,0)

    firstModel = predefinedModels.plummer{
        nbody       = model1Bodies,
        prng        = prng,
        position    = finalPosition,
        velocity    = finalVelocity,
        mass        = dwarfMass,
        scaleRadius = r0,
        ignore      = true
    }

      for i,v in ipairs(firstModel)
      do
     	v.ignore = false
      end

return firstModel
end

function makeHistogram()
   return HistogramParams.create()
end


