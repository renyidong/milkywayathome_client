/*
Copyright (C) 2011  Matthew Arsenault

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

#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

#include "nbody_types.h"
#include "milkyway_util.h"
#include "orbitintegrator.h"

#include "nbody_lua.h"
#include "nbody_lua_types.h"
#include "nbody_lua_models.h"
#include "lua_type_marshal.h"
#include "nbody_lua_marshal.h"
#include "nbody_check_params.h"

static int getNBodyCtxFunc(lua_State* luaSt)
{
    return mw_lua_checkglobalfunction(luaSt, "makeContext");
}

static int getNBodyPotentialFunc(lua_State* luaSt)
{
    return mw_lua_checkglobalfunction(luaSt, "makePotential");
}

static int getHistogramFunc(lua_State* luaSt)
{
    return mw_lua_checkglobalfunction(luaSt, "makeHistogram");
}

static int getBodiesFunc(lua_State* luaSt)
{
    return mw_lua_checkglobalfunction(luaSt, "makeBodies");
}

static void registerUsedStandardStuff(lua_State* luaSt)
{
    luaopen_base(luaSt);
    luaopen_table(luaSt);
    luaopen_string(luaSt);
    lua_pop(luaSt, 3);
}

static int bindServerArguments(lua_State* luaSt, const NBodyFlags* nbf)
{
    if (nbf->serverArgs)
        pushRealArray(luaSt, nbf->serverArgs, nbf->numServerArgs);
    else
        lua_pushnil(luaSt);
    lua_setglobal(luaSt, "serverArguments");

    if (nbf->setSeed)
        lua_pushinteger(luaSt, nbf->setSeed);
    else
        lua_pushnil(luaSt);
    lua_setglobal(luaSt, "serverSeed");

    return 0;
}

static lua_State* openNBodyLuaState(const NBodyFlags* nbf)
{
    char* script;
    lua_State* luaSt = NULL;

    script = mwReadFileResolved(nbf->inputFile);
    if (!script)
    {
        perror("Opening Lua script");
        return NULL;
    }

    luaSt = lua_open();
    if (!luaSt)
    {
        warn("Failed to get Lua state\n");
        free(script);
        return NULL;
    }

    registerUsedStandardStuff(luaSt);
    registerNBodyTypes(luaSt);
    registerOtherTypes(luaSt);
    registerPredefinedModelGenerators(luaSt);
    registerModelUtilityFunctions(luaSt);
    bindServerArguments(luaSt, nbf);

    if (luaL_dostring(luaSt, script))
    {
        mw_lua_pcall_warn(luaSt, "Error evaluating script");
        lua_close(luaSt);
        luaSt = NULL;
    }

    free(script);
    return luaSt;
}

static int evaluateContext(lua_State* luaSt, NBodyCtx* ctx)
{
    getNBodyCtxFunc(luaSt);
    if (lua_pcall(luaSt, 0, 1, 0))
    {
        mw_lua_pcall_warn(luaSt, "Error evaluating NBodyCtx");
        return 1;
    }

    *ctx = *checkNBodyCtx(luaSt, lua_gettop(luaSt));
    lua_pop(luaSt, 1);

    return contextSanityCheck(ctx);
}

static int evaluatePotential(lua_State* luaSt, Potential* pot)
{
    getNBodyPotentialFunc(luaSt);
    if (lua_pcall(luaSt, 0, 1, 0))
    {
        mw_lua_pcall_warn(luaSt, "Error evaluating Potential");
        return 1;
    }

    *pot = *checkPotential(luaSt, lua_gettop(luaSt));
    lua_pop(luaSt, 1);

    return potentialSanityCheck(pot);
}

static int evaluateHistogram(lua_State* luaSt, HistogramParams* hp)
{
    getHistogramFunc(luaSt);
    if (lua_pcall(luaSt, 0, 1, 0))
    {
        mw_lua_pcall_warn(luaSt, "Error evaluating HistogramParams");
        return 1;
    }

    *hp = *checkHistogramParams(luaSt, lua_gettop(luaSt));
    lua_pop(luaSt, 1);

    return 0;
}

static body* evaluateBodies(lua_State* luaSt, const NBodyCtx* ctx, const Potential* pot, unsigned int* n)
{
    int level, nResults;

    level = lua_gettop(luaSt);

    getBodiesFunc(luaSt);
    pushNBodyCtx(luaSt, ctx);
    pushPotential(luaSt, pot);

    if (lua_pcall(luaSt, 2, LUA_MULTRET, 0))
    {
        mw_lua_pcall_warn(luaSt, "Error evaluating bodies");
        return NULL;
    }

    nResults = lua_gettop(luaSt) - level;

    return readReturnedModels(luaSt, nResults, n);
}

static int setupInitialNBodyState(lua_State* luaSt, NBodyCtx* ctx, NBodyState* st)
{
    body* bodies;
    unsigned int n;

    if (evaluateContext(luaSt, ctx))
        return 1;

    if (evaluatePotential(luaSt, &ctx->pot))
        return 1;

    if (evaluateHistogram(luaSt, &ctx->histogramParams))
        return 1;

    bodies = evaluateBodies(luaSt, ctx, &ctx->pot, &n);
    if (!bodies)
        return 1;

    ctx->nbody = n;

    st->tree.rsize = ctx->treeRSize;
    st->tnow = 0.0;
    st->bodytab = bodies;

    return 0;
}

int setupNBody(NBodyCtx* ctx, NBodyState* st, const NBodyFlags* nbf)
{
    int rc;
    lua_State* luaSt;

    luaSt = openNBodyLuaState(nbf);
    if (!luaSt)
        return 1;

    rc = setupInitialNBodyState(luaSt, ctx, st);
    lua_close(luaSt);

    return rc;
}


