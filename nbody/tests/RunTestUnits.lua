
require "NBodyTesting"
require "persistence"

local arg = {...}

assert(#arg == 5, "Test driver expected 5 arguments got " .. #arg)

local nbodyBinary = arg[1]
local testDir = arg[2]
local testName = arg[3]
local histogramName = arg[4]
local testBodies = arg[5]

local nbodyFlags = getExtraNBodyFlags()
eprintf("NBODY_FLAGS = %s\n", nbodyFlags)

math.randomseed(os.time())

-- Pick one of the random seeds used in generating these tests
local testSeeds = { "123456789" }
local testSeed = testSeeds[math.random(1, #testSeeds)]

refResults = {
   ["model_1"] = {
      ["100"] = {
         ["123456789"] = 0.318025353945715
      },

      ["1024"] = {
         ["123456789"] = 1.487315346731191
      },

      ["10000"] = {
         ["123456789"] = 27.034203667790418
      }
   },

   ["model_2"] = {
      ["100"] = {
         ["123456789"] = 2.747461309670272
      },

      ["1024"] = {
         ["123456789"] = 73.450655727244310
      },

      ["10000"] = {
         ["123456789"] = 9999999.900000000372529
      }
   },

   ["model_3"] = {
      ["100"] = {
         ["123456789"] = 19.965972968107479
      },

      ["1024"] = {
         ["123456789"] = 311.757624580089498
      },

      ["10000"] = {
         ["123456789"] = 9999999.900000000372529
      }
   },

   ["model_4"] = {
      ["100"] = {
         ["123456789"] = 0.922724518138907
      },

      ["1024"] = {
         ["123456789"] = 5.808271252596239
      },

      ["10000"] = {
         ["123456789"] = 0.961795399390799
      }
   },

   ["model_5"] = {
      ["100"] = {
         ["123456789"] = 0.164613498208371
      },

      ["1024"] = {
         ["123456789"] = 4.914027899602377
      },

      ["10000"] = {
         ["123456789"] = 60.519248988925355
      }
   },

   ["model_5_bounds_test"] = {
      ["100"] = {
         ["123456789"] = 9999999.900000000372529
      },

      ["1024"] = {
         ["123456789"] = 9999999.900000000372529
      },

      ["10000"] = {
         ["123456789"] = 9999999.900000000372529
      }
   },

   ["model_6"] = {
      ["100"] = {
         ["123456789"] = 0.518738478423815
      },

      ["1024"] = {
         ["123456789"] = 7.176268186840729
      },

      ["10000"] = {
         ["123456789"] = 7.278506243994086
      }

   },

   ["model_7"] = {
      ["100"] = {
         ["123456789"] = 4.735612087125067
      },

      ["1024"] = {
         ["123456789"] = 5.894772574379092
      },

      ["10000"] = {
         ["123456789"] = 0.370491572410969
      }
   },

   ["model_triaxial"] = {
      ["100"] = {
         ["123456789"] = 0.093075941277474
      },

      ["1024"] = {
         ["123456789"] = 8.851608478333725
      },

      ["10000"] = {
         ["123456789"] = 91.806887790711215
      }
   }
}

function resultCloseEnough(a, b)
   return math.abs(a - b) < 1.0e-10
end

errFmtStr = [[
Result differs from expected:
   Expected = %20.15f  Actual = %20.15f  |Difference| = %20.15f
]]

function runCheckTest(testName, histogram, seed, nbody, ...)
   local fileResults, bodyResults
   local ret, result

   if not generatingResults then
      -- Check if the result exists first so we don't waste time on a useless test
      fileResults = assert(refResults[testName], "Didn't find result for test file")
      bodyResults = assert(fileResults[nbody], "Didn't find result with matching bodies")
      refResult = assert(bodyResults[seed], "Didn't find result with matching seed")
   end


   ret = runFullTest{
      nbodyBin  = nbodyBinary,
      testDir   = testDir,
      testName  = testName,
      histogram = histogram,
      seed      = seed,
      cached    = false,
      extraArgs = { nbody }
   }

   result = findLikelihood(ret)

   io.stdout:write(ret)

   if generatingResults then
      io.stderr:write(string.format("Test result: %d, %d, %s: %20.15f\n", nbody, seed, testName, result))
      return false
   end

   if result == nil then
      return true
   end

   local notClose = not resultCloseEnough(refResult, result)
   if notClose then
      io.stderr:write(string.format(errFmtStr, refResult, result, math.abs(result - refResult)))
   end

   return notClose
end

-- return true if passed
function testProbabilistic(resultFile, testName, histogram, nbody, iterations)
   local testTable, histTable, answer
   local resultTable = persisence.load(resultFile)
   assert(resultTable, "Failed to open result file " .. resultFile)

   testTable = assert(resultTable[testName], "Did not find result for test " .. testName)
   histTable = assert(testTable[nbody], "Did not find result for nbody " .. tostring(nbody))
   answer = assert(histTable[nbody], "Did not find result for histogram " .. histogram)

   local minAccepted = answer.mean - 3.0 * answer.stddev
   local maxAccepted = answer.mean + 3.0 * answer.stddev

   local result = 0.0
   local z = (result - answer.mean) / answer.stddev


   return true
end



function getResultName(testName)
   return string.format("%s__results.lua", testName)
end

if runCheckTest(testName, histogramName, testSeed, testBodies) then
   os.exit(1)
end
