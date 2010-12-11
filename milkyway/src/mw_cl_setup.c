/* Copyright 2010 Matthew Arsenault, Travis Desell, Boleslaw
Szymanski, Heidi Newberg, Carlos Varela, Malik Magdon-Ismail and
Rensselaer Polytechnic Institute.

This file is part of Milkway@Home.

Milkyway@Home is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Milkyway@Home is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Milkyway@Home.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>

#include "milkyway_cl.h"
#include "milkyway_util.h"
#include "milkyway_math.h"
#include "mw_cl_show_types.h"
#include "mw_cl_setup.h"
#include "mw_cl_program.h"
#include "mw_cl_device.h"
#include "mw_cl_util.h"

static cl_int mwCreateCtxQueue(CLInfo* ci, cl_bool useBufQueue)
{
    cl_int err = CL_SUCCESS;

    ci->clctx = clCreateContext(NULL, 1, &ci->dev, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
        mwCLWarn("Error creating context", err);
        return err;
    }

    ci->queue = clCreateCommandQueue(ci->clctx, ci->dev, 0, &err);
    if (err != CL_SUCCESS)
    {
        mwCLWarn("Error creating command queue", err);
        return err;
    }

    if (useBufQueue)
    {
        ci->bufQueue = clCreateCommandQueue(ci->clctx, ci->dev, 0, &err);
        if (err != CL_SUCCESS)
        {
            mwCLWarn("Error creating buffer command queue", err);
            return err;
        }
    }

    return CL_SUCCESS;
}

static cl_int mwGetCLInfo(CLInfo* ci, const CLRequest* clr)
{
    cl_int err = CL_SUCCESS;
    cl_uint n_platform;
    cl_platform_id* ids;
    cl_device_id* devs;
    cl_uint nDev;

    ids = mwGetAllPlatformIDs(ci, &n_platform);
    if (!ids)
    {
        warn("Failed to get any platforms\n");
        return MW_CL_ERROR;
    }

    mwPrintPlatforms(ids, n_platform);
    warn("Using device %u on platform %u\n", clr->devNum, clr->platform);

    devs = mwGetAllDevices(ids[clr->platform], &nDev);
    if (!devs)
    {
        warn("Error getting devices\n");
        free(ids);
        return MW_CL_ERROR;
    }

    err = mwSelectDevice(ci, devs, clr, nDev);
    if (err != CL_SUCCESS)
    {
        mwCLWarn("Failed to select a device", err);
        err = -1;
    }

    free(ids);
    free(devs);

    return err;
}


/* Query one device specified by type, create a context and command
 * queue, as well as retrieve device information
 */
cl_int mwSetupCL(CLInfo* ci,
                 DevInfo* di,   /* get detailed device information for selected device */
                 const CLRequest* clr)
{
    cl_int err;

    err = mwGetCLInfo(ci, clr);
    if (err != CL_SUCCESS)
    {
        warn("Failed to get information about device\n");
        return err;
    }

    err = mwGetDevInfo(di, ci->dev);
    if (err != CL_SUCCESS)
    {
        warn("Failed to get device info\n");
        return err;
    }

    mwPrintDevInfo(di);

  #if DOUBLEPREC
    if (!mwSupportsDoubles(di))
    {
        warn("Device doesn't support double precision\n");
        return MW_CL_ERROR;
    }
  #endif

    err = mwCreateCtxQueue(ci, CL_FALSE);
    if (err != CL_SUCCESS)
    {
        mwCLWarn("Error creating CL context and command queue", err);
        return err;
    }

    return CL_SUCCESS;
}

cl_int mwDestroyCLInfo(CLInfo* ci)
{
    cl_int err = CL_SUCCESS;
  /* Depending on where things fail, some of these will be NULL, and
   * will spew errors when trying to cleanup. */
    if (ci->queue)
        err |= clReleaseCommandQueue(ci->queue);
    if (ci->bufQueue)
        err |= clReleaseCommandQueue(ci->bufQueue);
    if (ci->prog)
        err |= clReleaseProgram(ci->prog);
    if (ci->kern)
        err |= clReleaseKernel(ci->kern);
    if (ci->clctx)
        err |= clReleaseContext(ci->clctx);

    /* TODO: or'ing the err and showing = useless */
    if (err)
        mwCLWarn("Error cleaning up CLInfo", err);

    return err;
}

