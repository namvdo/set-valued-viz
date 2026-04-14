let wasm;

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

let cachedTextDecoder = (typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true }) : { decode: () => { throw Error('TextDecoder not available') } } );

if (typeof TextDecoder !== 'undefined') { cachedTextDecoder.decode(); };

const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = (typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true }) : { decode: () => { throw Error('TextDecoder not available') } } );
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

const heap = new Array(128).fill(undefined);

heap.push(undefined, null, true, false);

let heap_next = heap.length;

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function getObject(idx) { return heap[idx]; }

let WASM_VECTOR_LEN = 0;

const cachedTextEncoder = (typeof TextEncoder !== 'undefined' ? new TextEncoder('utf-8') : { encode: () => { throw Error('TextEncoder not available') } } );

const encodeString = (typeof cachedTextEncoder.encodeInto === 'function'
    ? function (arg, view) {
    return cachedTextEncoder.encodeInto(arg, view);
}
    : function (arg, view) {
    const buf = cachedTextEncoder.encode(arg);
    view.set(buf);
    return {
        read: arg.length,
        written: buf.length
    };
});

function passStringToWasm0(arg, malloc, realloc) {

    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }

    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = encodeString(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedDataViewMemory0 = null;

function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        wasm.__wbindgen_exn_store(addHeapObject(e));
    }
}

function dropObject(idx) {
    if (idx < 132) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}
/**
 * @param {number} a
 * @param {number} b
 * @param {number} epsilon
 * @param {number} x_min
 * @param {number} x_max
 * @param {number} y_min
 * @param {number} y_max
 * @returns {any}
 */
export function compute_manifold_simple(a, b, epsilon, x_min, x_max, y_min, y_max) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.compute_manifold_simple(retptr, a, b, epsilon, x_min, x_max, y_min, y_max);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return takeObject(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * @param {number} a
 * @param {number} b
 * @param {number} epsilon
 * @param {number} saddle_x
 * @param {number} saddle_y
 * @param {number} period
 * @param {number} eigenvector_x
 * @param {number} eigenvector_y
 * @param {number} eigenvalue
 * @param {boolean} is_dual_repeller
 * @returns {any}
 */
export function compute_manifold_js(a, b, epsilon, saddle_x, saddle_y, period, eigenvector_x, eigenvector_y, eigenvalue, is_dual_repeller) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.compute_manifold_js(retptr, a, b, epsilon, saddle_x, saddle_y, period, eigenvector_x, eigenvector_y, eigenvalue, is_dual_repeller);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return takeObject(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * @param {string} x_eq
 * @param {string} y_eq
 * @param {any} params
 * @param {number} epsilon
 * @param {number} x_min
 * @param {number} x_max
 * @param {number} y_min
 * @param {number} y_max
 * @returns {any}
 */
export function compute_user_defined_manifold(x_eq, y_eq, params, epsilon, x_min, x_max, y_min, y_max) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passStringToWasm0(x_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(y_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.compute_user_defined_manifold(retptr, ptr0, len0, ptr1, len1, addHeapObject(params), epsilon, x_min, x_max, y_min, y_max);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return takeObject(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * orbits_js: Array of {points: [[x,y],...], period: number, stability: "stable"|"saddle"|"unstable"}
 * @param {number} a
 * @param {number} b
 * @param {number} epsilon
 * @param {number} x_min
 * @param {number} x_max
 * @param {number} y_min
 * @param {number} y_max
 * @param {any} orbits_js
 * @returns {any}
 */
export function compute_manifold_from_orbits(a, b, epsilon, x_min, x_max, y_min, y_max, orbits_js) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.compute_manifold_from_orbits(retptr, a, b, epsilon, x_min, x_max, y_min, y_max, addHeapObject(orbits_js));
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return takeObject(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * @param {number} x
 * @param {number} y
 * @param {string} x_eq
 * @param {string} y_eq
 * @param {any} params
 * @param {number} epsilon
 * @returns {any}
 */
export function evaluate_user_defined_map(x, y, x_eq, y_eq, params, epsilon) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passStringToWasm0(x_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(y_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.evaluate_user_defined_map(retptr, x, y, ptr0, len0, ptr1, len1, addHeapObject(params), epsilon);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return takeObject(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * @param {number} a
 * @param {number} b
 * @param {number} epsilon
 * @param {number} x_min
 * @param {number} x_max
 * @param {number} y_min
 * @param {number} y_max
 * @param {any} orbits_js
 * @param {number} intersection_threshold
 * @returns {any}
 */
export function compute_stable_and_unstable_manifolds(a, b, epsilon, x_min, x_max, y_min, y_max, orbits_js, intersection_threshold) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.compute_stable_and_unstable_manifolds(retptr, a, b, epsilon, x_min, x_max, y_min, y_max, addHeapObject(orbits_js), intersection_threshold);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return takeObject(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * @param {string} x_eq
 * @param {string} y_eq
 * @param {any} params
 * @param {number} epsilon
 * @param {number} x_min
 * @param {number} x_max
 * @param {number} y_min
 * @param {number} y_max
 * @param {any} orbits_js
 * @returns {any}
 */
export function compute_manifold_from_orbits_user_defined(x_eq, y_eq, params, epsilon, x_min, x_max, y_min, y_max, orbits_js) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passStringToWasm0(x_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(y_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.compute_manifold_from_orbits_user_defined(retptr, ptr0, len0, ptr1, len1, addHeapObject(params), epsilon, x_min, x_max, y_min, y_max, addHeapObject(orbits_js));
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return takeObject(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * @param {string} x_eq
 * @param {string} y_eq
 * @param {any} params
 * @param {number} epsilon
 * @param {number} x_min
 * @param {number} x_max
 * @param {number} y_min
 * @param {number} y_max
 * @param {any} orbits_js
 * @param {number} intersection_threshold
 * @returns {any}
 */
export function compute_stable_and_unstable_manifolds_user_defined(x_eq, y_eq, params, epsilon, x_min, x_max, y_min, y_max, orbits_js, intersection_threshold) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passStringToWasm0(x_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(y_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.compute_stable_and_unstable_manifolds_user_defined(retptr, ptr0, len0, ptr1, len1, addHeapObject(params), epsilon, x_min, x_max, y_min, y_max, addHeapObject(orbits_js), intersection_threshold);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return takeObject(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * @param {number} x
 * @param {number} y
 * @param {number} nx
 * @param {number} ny
 * @param {number} delta
 * @param {number} h
 * @param {number} epsilon
 * @returns {any}
 */
export function boundary_map_duffing_ode(x, y, nx, ny, delta, h, epsilon) {
    const ret = wasm.boundary_map_duffing_ode(x, y, nx, ny, delta, h, epsilon);
    return takeObject(ret);
}

/**
 * @param {number} x
 * @param {number} y
 * @param {string} x_eq
 * @param {string} y_eq
 * @param {any} params
 * @returns {any}
 */
export function evaluate_user_defined_ode(x, y, x_eq, y_eq, params) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passStringToWasm0(x_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(y_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.evaluate_user_defined_ode(retptr, x, y, ptr0, len0, ptr1, len1, addHeapObject(params));
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return takeObject(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * @param {number} x
 * @param {number} y
 * @param {number} nx
 * @param {number} ny
 * @param {string} x_eq
 * @param {string} y_eq
 * @param {any} params
 * @param {number} h
 * @param {number} epsilon
 * @returns {any}
 */
export function boundary_map_user_defined_ode(x, y, nx, ny, x_eq, y_eq, params, h, epsilon) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passStringToWasm0(x_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(y_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.boundary_map_user_defined_ode(retptr, x, y, nx, ny, ptr0, len0, ptr1, len1, addHeapObject(params), h, epsilon);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return takeObject(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * @param {number} x
 * @param {number} y
 * @param {number} nx
 * @param {number} ny
 * @param {string} x_eq
 * @param {string} y_eq
 * @param {any} params
 * @param {number} epsilon
 * @returns {any}
 */
export function boundary_map_user_defined(x, y, nx, ny, x_eq, y_eq, params, epsilon) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passStringToWasm0(x_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(y_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.boundary_map_user_defined(retptr, x, y, nx, ny, ptr0, len0, ptr1, len1, addHeapObject(params), epsilon);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return takeObject(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * Legacy Hénon-specific sweep (kept for backward compat, routes through generic pipeline)
 * @param {number} b
 * @param {number} epsilon
 * @param {number} a_min
 * @param {number} a_max
 * @param {number} num_samples
 * @param {number} max_period
 * @param {number} x_min
 * @param {number} x_max
 * @param {number} y_min
 * @param {number} y_max
 * @returns {any}
 */
export function parameterSweep(b, epsilon, a_min, a_max, num_samples, max_period, x_min, x_max, y_min, y_max) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.parameterSweep(retptr, b, epsilon, a_min, a_max, num_samples, max_period, x_min, x_max, y_min, y_max);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return takeObject(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * Unified parameter sweep: works for any system type + any parameter.
 * @param {string} system_type
 * @param {string} x_eq
 * @param {string} y_eq
 * @param {any} params_js
 * @param {string} sweep_param_name
 * @param {number} sweep_min
 * @param {number} sweep_max
 * @param {number} num_samples
 * @param {number} epsilon
 * @param {number} max_period
 * @param {number} x_min
 * @param {number} x_max
 * @param {number} y_min
 * @param {number} y_max
 * @returns {any}
 */
export function parameterSweepGeneric(system_type, x_eq, y_eq, params_js, sweep_param_name, sweep_min, sweep_max, num_samples, epsilon, max_period, x_min, x_max, y_min, y_max) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passStringToWasm0(system_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(x_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(y_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(sweep_param_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len3 = WASM_VECTOR_LEN;
        wasm.parameterSweepGeneric(retptr, ptr0, len0, ptr1, len1, ptr2, len2, addHeapObject(params_js), ptr3, len3, sweep_min, sweep_max, num_samples, epsilon, max_period, x_min, x_max, y_min, y_max);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return takeObject(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * @param {number} b
 * @param {number} epsilon
 * @param {number} a_min
 * @param {number} a_max
 * @param {number} num_samples
 * @param {number} max_period
 * @param {number} x_min
 * @param {number} x_max
 * @param {number} y_min
 * @param {number} y_max
 * @returns {string}
 */
export function parameterSweepCsv(b, epsilon, a_min, a_max, num_samples, max_period, x_min, x_max, y_min, y_max) {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.parameterSweepCsv(retptr, b, epsilon, a_min, a_max, num_samples, max_period, x_min, x_max, y_min, y_max);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @param {number} a
 * @param {number} b
 * @param {number} epsilon
 * @param {number} x_min
 * @param {number} x_max
 * @param {number} y_min
 * @param {number} y_max
 * @returns {any}
 */
export function compute_duffing_manifold_simple(a, b, epsilon, x_min, x_max, y_min, y_max) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.compute_duffing_manifold_simple(retptr, a, b, epsilon, x_min, x_max, y_min, y_max);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return takeObject(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}
/**
 * @param {any} unstable_plus_js
 * @param {any} unstable_minus_js
 * @param {any} stable_plus_js
 * @param {any} stable_minus_js
 * @param {number} num_closest_pairs
 * @returns {any}
 */
export function compute_hausdorff_distance_between_manifolds(unstable_plus_js, unstable_minus_js, stable_plus_js, stable_minus_js, num_closest_pairs) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.compute_hausdorff_distance_between_manifolds(retptr, addHeapObject(unstable_plus_js), addHeapObject(unstable_minus_js), addHeapObject(stable_plus_js), addHeapObject(stable_minus_js), num_closest_pairs);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return takeObject(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * @param {number} b
 * @param {number} epsilon
 * @param {number} a_min
 * @param {number} a_max
 * @param {number} num_samples
 * @returns {any}
 */
export function compute_bifurcation_hausdorff(b, epsilon, a_min, a_max, num_samples) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.compute_bifurcation_hausdorff(retptr, b, epsilon, a_min, a_max, num_samples);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return takeObject(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

/**
 * @enum {0 | 1 | 2}
 */
export const DuffingStabilityType = Object.freeze({
    Stable: 0, "0": "Stable",
    Unstable: 1, "1": "Unstable",
    Saddle: 2, "2": "Saddle",
});
/**
 * @enum {0 | 1 | 2}
 */
export const PeriodicType = Object.freeze({
    Stable: 0, "0": "Stable",
    Unstable: 1, "1": "Unstable",
    Saddle: 2, "2": "Saddle",
});
/**
 * @enum {0 | 1 | 2 | 3 | 4}
 */
export const RecordingStatus = Object.freeze({
    Idle: 0, "0": "Idle",
    Recording: 1, "1": "Recording",
    Encoding: 2, "2": "Encoding",
    Complete: 3, "3": "Complete",
    Error: 4, "4": "Error",
});
/**
 * @enum {0 | 1 | 2}
 */
export const StabilityType = Object.freeze({
    Stable: 0, "0": "Stable",
    Unstable: 1, "1": "Unstable",
    Saddle: 2, "2": "Saddle",
});

const BdeSimulatorUserDefinedWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_bdesimulatoruserdefinedwasm_free(ptr >>> 0, 1));

export class BdeSimulatorUserDefinedWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BdeSimulatorUserDefinedWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_bdesimulatoruserdefinedwasm_free(ptr, 0);
    }
    /**
     * @param {string} x_eq
     * @param {string} y_eq
     * @param {any} params
     * @param {number} epsilon
     * @param {number} cx
     * @param {number} cy
     * @param {number} r
     * @param {number} num_points
     */
    constructor(x_eq, y_eq, params, epsilon, cx, cy, r, num_points) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(x_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passStringToWasm0(y_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            wasm.bdesimulatoruserdefinedwasm_new(retptr, ptr0, len0, ptr1, len1, addHeapObject(params), epsilon, cx, cy, r, num_points);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            BdeSimulatorUserDefinedWasmFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} h
     * @returns {any}
     */
    step(h) {
        const ret = wasm.bdesimulatoruserdefinedwasm_step(this.__wbg_ptr, h);
        return takeObject(ret);
    }
    /**
     * @returns {any}
     */
    get_points() {
        const ret = wasm.bdesimulatoruserdefinedwasm_get_points(this.__wbg_ptr);
        return takeObject(ret);
    }
    reparameterize() {
        wasm.bdesimulatoruserdefinedwasm_reparameterize(this.__wbg_ptr);
    }
    /**
     * @param {number} gap
     * @returns {number}
     */
    has_self_intersection(gap) {
        const ret = wasm.bdesimulatoruserdefinedwasm_has_self_intersection(this.__wbg_ptr, gap);
        return ret;
    }
    /**
     * @param {number} speed_threshold
     * @returns {any}
     */
    get_fold_indices(speed_threshold) {
        const ret = wasm.bdesimulatoruserdefinedwasm_get_fold_indices(this.__wbg_ptr, speed_threshold);
        return takeObject(ret);
    }
}

const BdeSimulatorWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_bdesimulatorwasm_free(ptr >>> 0, 1));

export class BdeSimulatorWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BdeSimulatorWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_bdesimulatorwasm_free(ptr, 0);
    }
    /**
     * @param {number} delta
     * @param {number} epsilon
     * @param {number} cx
     * @param {number} cy
     * @param {number} r
     * @param {number} num_points
     */
    constructor(delta, epsilon, cx, cy, r, num_points) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.bdesimulatorwasm_new(retptr, delta, epsilon, cx, cy, r, num_points);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            BdeSimulatorWasmFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} h
     * @returns {any}
     */
    step(h) {
        const ret = wasm.bdesimulatorwasm_step(this.__wbg_ptr, h);
        return takeObject(ret);
    }
    /**
     * @returns {any}
     */
    get_points() {
        const ret = wasm.bdesimulatorwasm_get_points(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * arc-length reparameterize: redistribute points evenly along the curve.
     */
    reparameterize() {
        wasm.bdesimulatorwasm_reparameterize(this.__wbg_ptr);
    }
    /**
     * @param {number} gap
     * @returns {number}
     */
    has_self_intersection(gap) {
        const ret = wasm.bdesimulatorwasm_has_self_intersection(this.__wbg_ptr, gap);
        return ret;
    }
    /**
     * @param {number} speed_threshold
     * @returns {any}
     */
    get_fold_indices(speed_threshold) {
        const ret = wasm.bdesimulatorwasm_get_fold_indices(this.__wbg_ptr, speed_threshold);
        return takeObject(ret);
    }
}

const BoundaryHenonSystemWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_boundaryhenonsystemwasm_free(ptr >>> 0, 1));

export class BoundaryHenonSystemWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BoundaryHenonSystemWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_boundaryhenonsystemwasm_free(ptr, 0);
    }
    /**
     * @param {number} a
     * @param {number} b
     * @param {number} epsilon
     * @param {number} max_period
     * @param {number} x_min
     * @param {number} x_max
     * @param {number} y_min
     * @param {number} y_max
     */
    constructor(a, b, epsilon, max_period, x_min, x_max, y_min, y_max) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.boundaryhenonsystemwasm_new(retptr, a, b, epsilon, max_period, x_min, x_max, y_min, y_max);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            BoundaryHenonSystemWasmFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {any}
     */
    getPeriodicOrbits() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.boundaryhenonsystemwasm_getPeriodicOrbits(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} initial_x
     * @param {number} initial_y
     * @param {number} initial_nx
     * @param {number} initial_ny
     * @param {number} max_iterations
     */
    trackTrajectory(initial_x, initial_y, initial_nx, initial_ny, max_iterations) {
        wasm.boundaryhenonsystemwasm_trackTrajectory(this.__wbg_ptr, initial_x, initial_y, initial_nx, initial_ny, max_iterations);
    }
    /**
     * @returns {any}
     */
    getCurrentPoint() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.boundaryhenonsystemwasm_getCurrentPoint(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} start
     * @param {number} end
     * @returns {any}
     */
    getTrajectory(start, end) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.boundaryhenonsystemwasm_getTrajectory(retptr, this.__wbg_ptr, start, end);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {boolean}
     */
    step() {
        const ret = wasm.boundaryhenonsystemwasm_step(this.__wbg_ptr);
        return ret !== 0;
    }
    reset() {
        wasm.boundaryhenonsystemwasm_reset(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    getTotalIterations() {
        const ret = wasm.boundaryhenonsystemwasm_getTotalIterations(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    getCurrentIteration() {
        const ret = wasm.boundaryhenonsystemwasm_getCurrentIteration(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    getOrbitCount() {
        const ret = wasm.boundaryhenonsystemwasm_getOrbitCount(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    getEpsilon() {
        const ret = wasm.boundaryhenonsystemwasm_getEpsilon(this.__wbg_ptr);
        return ret;
    }
}

const BoundaryUserDefinedSystemWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_boundaryuserdefinedsystemwasm_free(ptr >>> 0, 1));

export class BoundaryUserDefinedSystemWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BoundaryUserDefinedSystemWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_boundaryuserdefinedsystemwasm_free(ptr, 0);
    }
    /**
     * @param {string} x_eq
     * @param {string} y_eq
     * @param {any} params
     * @param {number} epsilon
     * @param {number} max_period
     * @param {number} x_min
     * @param {number} x_max
     * @param {number} y_min
     * @param {number} y_max
     */
    constructor(x_eq, y_eq, params, epsilon, max_period, x_min, x_max, y_min, y_max) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(x_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passStringToWasm0(y_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            wasm.boundaryuserdefinedsystemwasm_new(retptr, ptr0, len0, ptr1, len1, addHeapObject(params), epsilon, max_period, x_min, x_max, y_min, y_max);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            BoundaryUserDefinedSystemWasmFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {any}
     */
    getPeriodicOrbits() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.boundaryuserdefinedsystemwasm_getPeriodicOrbits(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {number}
     */
    getOrbitCount() {
        const ret = wasm.boundaryuserdefinedsystemwasm_getOrbitCount(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    getEpsilon() {
        const ret = wasm.boundaryuserdefinedsystemwasm_getEpsilon(this.__wbg_ptr);
        return ret;
    }
}

const DuffingParamsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_duffingparams_free(ptr >>> 0, 1));

export class DuffingParams {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DuffingParamsFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_duffingparams_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get a() {
        const ret = wasm.__wbg_get_duffingparams_a(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set a(arg0) {
        wasm.__wbg_set_duffingparams_a(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get b() {
        const ret = wasm.__wbg_get_duffingparams_b(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set b(arg0) {
        wasm.__wbg_set_duffingparams_b(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get epsilon() {
        const ret = wasm.__wbg_get_duffingparams_epsilon(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set epsilon(arg0) {
        wasm.__wbg_set_duffingparams_epsilon(this.__wbg_ptr, arg0);
    }
}

const DuffingSystemWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_duffingsystemwasm_free(ptr >>> 0, 1));

export class DuffingSystemWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DuffingSystemWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_duffingsystemwasm_free(ptr, 0);
    }
    /**
     * @param {number} a
     * @param {number} b
     * @param {number} max_period
     */
    constructor(a, b, max_period) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.duffingsystemwasm_new(retptr, a, b, max_period);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            DuffingSystemWasmFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {any}
     */
    getPeriodicOrbits() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.duffingsystemwasm_getPeriodicOrbits(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} initial_x
     * @param {number} initial_y
     * @param {number} max_iterations
     */
    trackTrajectory(initial_x, initial_y, max_iterations) {
        wasm.duffingsystemwasm_trackTrajectory(this.__wbg_ptr, initial_x, initial_y, max_iterations);
    }
    /**
     * @returns {any}
     */
    getCurrentPoint() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.duffingsystemwasm_getCurrentPoint(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} start
     * @param {number} end
     * @returns {any}
     */
    getTrajectory(start, end) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.duffingsystemwasm_getTrajectory(retptr, this.__wbg_ptr, start, end);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {boolean}
     */
    step() {
        const ret = wasm.duffingsystemwasm_step(this.__wbg_ptr);
        return ret !== 0;
    }
    reset() {
        wasm.duffingsystemwasm_reset(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    getTotalIterations() {
        const ret = wasm.duffingsystemwasm_getTotalIterations(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    getCurrentIteration() {
        const ret = wasm.duffingsystemwasm_getCurrentIteration(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    getOrbitCount() {
        const ret = wasm.duffingsystemwasm_getOrbitCount(this.__wbg_ptr);
        return ret >>> 0;
    }
}

const EulerMapSystemWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_eulermapsystemwasm_free(ptr >>> 0, 1));

export class EulerMapSystemWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EulerMapSystemWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_eulermapsystemwasm_free(ptr, 0);
    }
    /**
     * @param {number} delta
     * @param {number} h
     * @param {number} epsilon
     * @param {number} max_period
     */
    constructor(delta, h, epsilon, max_period) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.eulermapsystemwasm_new(retptr, delta, h, epsilon, max_period);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            EulerMapSystemWasmFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {any}
     */
    getPeriodicOrbits() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.eulermapsystemwasm_getPeriodicOrbits(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} _initial_x
     * @param {number} _initial_y
     * @param {number} _max_iterations
     */
    trackTrajectory(_initial_x, _initial_y, _max_iterations) {
        wasm.eulermapsystemwasm_trackTrajectory(this.__wbg_ptr, _initial_x, _initial_y, _max_iterations);
    }
    /**
     * @returns {any}
     */
    getCurrentPoint() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.eulermapsystemwasm_getCurrentPoint(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} _start
     * @param {number} _end
     * @returns {any}
     */
    getTrajectory(_start, _end) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.eulermapsystemwasm_getTrajectory(retptr, this.__wbg_ptr, _start, _end);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {boolean}
     */
    step() {
        const ret = wasm.eulermapsystemwasm_step(this.__wbg_ptr);
        return ret !== 0;
    }
    reset() {
        wasm.eulermapsystemwasm_reset(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    getTotalIterations() {
        const ret = wasm.eulermapsystemwasm_getCurrentIteration(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    getCurrentIteration() {
        const ret = wasm.eulermapsystemwasm_getCurrentIteration(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    getOrbitCount() {
        const ret = wasm.eulermapsystemwasm_getCurrentIteration(this.__wbg_ptr);
        return ret >>> 0;
    }
}

const ExtendedPointFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_extendedpoint_free(ptr >>> 0, 1));

export class ExtendedPoint {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ExtendedPointFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_extendedpoint_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get x() {
        const ret = wasm.__wbg_get_extendedpoint_x(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set x(arg0) {
        wasm.__wbg_set_extendedpoint_x(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get y() {
        const ret = wasm.__wbg_get_extendedpoint_y(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set y(arg0) {
        wasm.__wbg_set_extendedpoint_y(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get nx() {
        const ret = wasm.__wbg_get_extendedpoint_nx(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set nx(arg0) {
        wasm.__wbg_set_extendedpoint_nx(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get ny() {
        const ret = wasm.__wbg_get_extendedpoint_ny(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set ny(arg0) {
        wasm.__wbg_set_extendedpoint_ny(this.__wbg_ptr, arg0);
    }
}

const HenonParamsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_henonparams_free(ptr >>> 0, 1));

export class HenonParams {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        HenonParamsFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_henonparams_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get a() {
        const ret = wasm.__wbg_get_henonparams_a(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set a(arg0) {
        wasm.__wbg_set_henonparams_a(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get b() {
        const ret = wasm.__wbg_get_henonparams_b(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set b(arg0) {
        wasm.__wbg_set_henonparams_b(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get epsilon() {
        const ret = wasm.__wbg_get_henonparams_epsilon(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set epsilon(arg0) {
        wasm.__wbg_set_henonparams_epsilon(this.__wbg_ptr, arg0);
    }
}

const UlamComputerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_ulamcomputer_free(ptr >>> 0, 1));
/**
 * UlamComputer computes the transition matrix and invariant measures
 * using the Ulam/GAIO method with epsilon-inflation
 */
export class UlamComputer {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        UlamComputerFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_ulamcomputer_free(ptr, 0);
    }
    /**
     * Create a new UlamComputer with the given parameters
     *
     * # Arguments
     * * `a` - Henon map parameter a
     * * `b` - Henon map parameter b
     * * `subdivisions` - Number of grid subdivisions in each dimension
     * * `points_per_box` - Number of sample points per box (will be squared for grid)
     * * `epsilon` - Epsilon parameter for ball inflation (boundary detection)
     * @param {number} a
     * @param {number} b
     * @param {number} subdivisions
     * @param {number} points_per_box
     * @param {number} epsilon
     * @param {number} x_min
     * @param {number} x_max
     * @param {number} y_min
     * @param {number} y_max
     */
    constructor(a, b, subdivisions, points_per_box, epsilon, x_min, x_max, y_min, y_max) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.ulamcomputer_new(retptr, a, b, subdivisions, points_per_box, epsilon, x_min, x_max, y_min, y_max);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            UlamComputerFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Get the grid boxes as a serialized array
     * @returns {any}
     */
    get_grid_boxes() {
        const ret = wasm.ulamcomputer_get_grid_boxes(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @param {number} from_box_idx
     * @returns {any}
     */
    get_transitions(from_box_idx) {
        const ret = wasm.ulamcomputer_get_transitions(this.__wbg_ptr, from_box_idx);
        return takeObject(ret);
    }
    /**
     * Get the right eigenvector (invariant measure, forward dynamics)
     * @returns {any}
     */
    get_invariant_measure() {
        const ret = wasm.ulamcomputer_get_invariant_measure(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * Get the left eigenvector (backward invariant measure)
     * @returns {any}
     */
    get_left_eigenvector() {
        const ret = wasm.ulamcomputer_get_left_eigenvector(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * Get the epsilon parameter used for this computation
     * @returns {number}
     */
    get_epsilon() {
        const ret = wasm.ulamcomputer_get_epsilon(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the grid step size (useful for UI scaling)
     * @returns {any}
     */
    get_grid_step() {
        const ret = wasm.ulamcomputer_get_grid_step(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @param {number} x
     * @param {number} y
     * @returns {number}
     */
    get_box_index(x, y) {
        const ret = wasm.ulamcomputer_get_box_index(this.__wbg_ptr, x, y);
        return ret;
    }
    /**
     * @param {number} x
     * @param {number} y
     * @returns {any}
     */
    get_intersecting_boxes(x, y) {
        const ret = wasm.ulamcomputer_get_intersecting_boxes(this.__wbg_ptr, x, y);
        return takeObject(ret);
    }
    /**
     * @returns {any}
     */
    get_dimensions() {
        const ret = wasm.ulamcomputer_get_dimensions(this.__wbg_ptr);
        return takeObject(ret);
    }
}

const UlamComputerContinuousFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_ulamcomputercontinuous_free(ptr >>> 0, 1));

export class UlamComputerContinuous {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        UlamComputerContinuousFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_ulamcomputercontinuous_free(ptr, 0);
    }
    /**
     * Build the Ulam matrix for the Duffing ODE  ẋ=y, ẏ=x−x³−δy
     * using the time-T flow map as the generating discrete map.
     *
     * Arguments
     * * `delta`        – damping δ
     * * `capital_t`    – integration time T per discrete step
     * * `subdivisions` – grid cells per axis
     * * `points_per_box` – sample density
     * * `epsilon`      – epsilon ball inflation for set-valued images
     * @param {number} delta
     * @param {number} capital_t
     * @param {number} subdivisions
     * @param {number} points_per_box
     * @param {number} epsilon
     * @param {number} x_min
     * @param {number} x_max
     * @param {number} y_min
     * @param {number} y_max
     */
    constructor(delta, capital_t, subdivisions, points_per_box, epsilon, x_min, x_max, y_min, y_max) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.ulamcomputercontinuous_new(retptr, delta, capital_t, subdivisions, points_per_box, epsilon, x_min, x_max, y_min, y_max);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            UlamComputerContinuousFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {any}
     */
    get_grid_boxes() {
        const ret = wasm.ulamcomputercontinuous_get_grid_boxes(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @param {number} from_box_idx
     * @returns {any}
     */
    get_transitions(from_box_idx) {
        const ret = wasm.ulamcomputercontinuous_get_transitions(this.__wbg_ptr, from_box_idx);
        return takeObject(ret);
    }
    /**
     * @returns {any}
     */
    get_invariant_measure() {
        const ret = wasm.ulamcomputercontinuous_get_invariant_measure(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @returns {any}
     */
    get_left_eigenvector() {
        const ret = wasm.ulamcomputercontinuous_get_left_eigenvector(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @returns {number}
     */
    get_epsilon() {
        const ret = wasm.ulamcomputercontinuous_get_epsilon(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get_grid_step() {
        const ret = wasm.ulamcomputercontinuous_get_grid_step(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @returns {any}
     */
    get_dimensions() {
        const ret = wasm.ulamcomputercontinuous_get_dimensions(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @param {number} x
     * @param {number} y
     * @returns {number}
     */
    get_box_index(x, y) {
        const ret = wasm.ulamcomputercontinuous_get_box_index(this.__wbg_ptr, x, y);
        return ret;
    }
}

const UlamComputerContinuousUserDefinedFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_ulamcomputercontinuoususerdefined_free(ptr >>> 0, 1));

export class UlamComputerContinuousUserDefined {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        UlamComputerContinuousUserDefinedFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_ulamcomputercontinuoususerdefined_free(ptr, 0);
    }
    /**
     * Build the Ulam matrix for a user-defined ODE using the time-T flow map.
     *
     * Arguments
     * * `x_eq`, `y_eq` – vector field components ẋ, ẏ
     * * `params`      – parameter list (name/value)
     * * `capital_t`   – integration time T per discrete step
     * * `subdivisions` – grid cells per axis
     * * `points_per_box` – sample density
     * * `epsilon`     – epsilon ball inflation for set-valued images
     * @param {string} x_eq
     * @param {string} y_eq
     * @param {any} params
     * @param {number} capital_t
     * @param {number} subdivisions
     * @param {number} points_per_box
     * @param {number} epsilon
     * @param {number} x_min
     * @param {number} x_max
     * @param {number} y_min
     * @param {number} y_max
     */
    constructor(x_eq, y_eq, params, capital_t, subdivisions, points_per_box, epsilon, x_min, x_max, y_min, y_max) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(x_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passStringToWasm0(y_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            wasm.ulamcomputercontinuoususerdefined_new(retptr, ptr0, len0, ptr1, len1, addHeapObject(params), capital_t, subdivisions, points_per_box, epsilon, x_min, x_max, y_min, y_max);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            UlamComputerContinuousUserDefinedFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {any}
     */
    get_grid_boxes() {
        const ret = wasm.ulamcomputercontinuoususerdefined_get_grid_boxes(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @param {number} from_box_idx
     * @returns {any}
     */
    get_transitions(from_box_idx) {
        const ret = wasm.ulamcomputercontinuoususerdefined_get_transitions(this.__wbg_ptr, from_box_idx);
        return takeObject(ret);
    }
    /**
     * @returns {any}
     */
    get_invariant_measure() {
        const ret = wasm.ulamcomputercontinuoususerdefined_get_invariant_measure(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @returns {any}
     */
    get_left_eigenvector() {
        const ret = wasm.ulamcomputercontinuoususerdefined_get_left_eigenvector(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @returns {number}
     */
    get_epsilon() {
        const ret = wasm.ulamcomputercontinuoususerdefined_get_epsilon(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get_grid_step() {
        const ret = wasm.ulamcomputercontinuoususerdefined_get_grid_step(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @returns {any}
     */
    get_dimensions() {
        const ret = wasm.ulamcomputercontinuoususerdefined_get_dimensions(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @param {number} x
     * @param {number} y
     * @returns {number}
     */
    get_box_index(x, y) {
        const ret = wasm.ulamcomputercontinuoususerdefined_get_box_index(this.__wbg_ptr, x, y);
        return ret;
    }
}

const UlamComputerUserDefinedFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_ulamcomputeruserdefined_free(ptr >>> 0, 1));

export class UlamComputerUserDefined {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        UlamComputerUserDefinedFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_ulamcomputeruserdefined_free(ptr, 0);
    }
    /**
     * @param {string} x_eq
     * @param {string} y_eq
     * @param {any} params
     * @param {number} subdivisions
     * @param {number} points_per_box
     * @param {number} epsilon
     * @param {number} x_min
     * @param {number} x_max
     * @param {number} y_min
     * @param {number} y_max
     */
    constructor(x_eq, y_eq, params, subdivisions, points_per_box, epsilon, x_min, x_max, y_min, y_max) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(x_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passStringToWasm0(y_eq, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            wasm.ulamcomputeruserdefined_new(retptr, ptr0, len0, ptr1, len1, addHeapObject(params), subdivisions, points_per_box, epsilon, x_min, x_max, y_min, y_max);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            UlamComputerUserDefinedFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {any}
     */
    get_grid_boxes() {
        const ret = wasm.ulamcomputeruserdefined_get_grid_boxes(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @param {number} from_box_idx
     * @returns {any}
     */
    get_transitions(from_box_idx) {
        const ret = wasm.ulamcomputeruserdefined_get_transitions(this.__wbg_ptr, from_box_idx);
        return takeObject(ret);
    }
    /**
     * @returns {any}
     */
    get_invariant_measure() {
        const ret = wasm.ulamcomputeruserdefined_get_invariant_measure(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @returns {any}
     */
    get_left_eigenvector() {
        const ret = wasm.ulamcomputeruserdefined_get_left_eigenvector(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @returns {number}
     */
    get_epsilon() {
        const ret = wasm.ulamcomputeruserdefined_get_epsilon(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {any}
     */
    get_grid_step() {
        const ret = wasm.ulamcomputeruserdefined_get_grid_step(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @param {number} x
     * @param {number} y
     * @returns {number}
     */
    get_box_index(x, y) {
        const ret = wasm.ulamcomputeruserdefined_get_box_index(this.__wbg_ptr, x, y);
        return ret;
    }
    /**
     * @param {number} x
     * @param {number} y
     * @returns {any}
     */
    get_intersecting_boxes(x, y) {
        const ret = wasm.ulamcomputeruserdefined_get_intersecting_boxes(this.__wbg_ptr, x, y);
        return takeObject(ret);
    }
    /**
     * @returns {any}
     */
    get_dimensions() {
        const ret = wasm.ulamcomputeruserdefined_get_dimensions(this.__wbg_ptr);
        return takeObject(ret);
    }
}

const VideoConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_videoconfig_free(ptr >>> 0, 1));

export class VideoConfig {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(VideoConfig.prototype);
        obj.__wbg_ptr = ptr;
        VideoConfigFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        VideoConfigFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_videoconfig_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get width() {
        const ret = wasm.__wbg_get_videoconfig_width(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set width(arg0) {
        wasm.__wbg_set_videoconfig_width(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get height() {
        const ret = wasm.__wbg_get_videoconfig_height(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set height(arg0) {
        wasm.__wbg_set_videoconfig_height(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get fps() {
        const ret = wasm.__wbg_get_videoconfig_fps(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set fps(arg0) {
        wasm.__wbg_set_videoconfig_fps(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get crf() {
        const ret = wasm.__wbg_get_videoconfig_crf(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set crf(arg0) {
        wasm.__wbg_set_videoconfig_crf(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} width
     * @param {number} height
     * @param {number} fps
     * @param {number} crf
     */
    constructor(width, height, fps, crf) {
        const ret = wasm.videoconfig_new(width, height, fps, crf);
        this.__wbg_ptr = ret >>> 0;
        VideoConfigFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {VideoConfig}
     */
    static default_config() {
        const ret = wasm.videoconfig_default_config();
        return VideoConfig.__wrap(ret);
    }
}

const VideoRecorderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_videorecorder_free(ptr >>> 0, 1));
/**
 * Video recorder state machine for coordinating frame capture
 */
export class VideoRecorder {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        VideoRecorderFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_videorecorder_free(ptr, 0);
    }
    constructor() {
        const ret = wasm.videorecorder_new();
        this.__wbg_ptr = ret >>> 0;
        VideoRecorderFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Start recording with current parameters
     * @param {number} a
     * @param {number} b
     * @param {number} epsilon
     * @param {string} animated_param
     * @param {number} range_start
     * @param {number} range_end
     * @returns {boolean}
     */
    start_recording(a, b, epsilon, animated_param, range_start, range_end) {
        const ptr0 = passStringToWasm0(animated_param, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.videorecorder_start_recording(this.__wbg_ptr, a, b, epsilon, ptr0, len0, range_start, range_end);
        return ret !== 0;
    }
    /**
     * Record a frame timestamp
     * @param {number} _parameter_value
     * @returns {number}
     */
    add_frame(_parameter_value) {
        const ret = wasm.videorecorder_add_frame(this.__wbg_ptr, _parameter_value);
        return ret >>> 0;
    }
    /**
     * Get current frame count
     * @returns {number}
     */
    get_frame_count() {
        const ret = wasm.videorecorder_get_frame_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get recording status
     * @returns {RecordingStatus}
     */
    get_status() {
        const ret = wasm.videorecorder_get_status(this.__wbg_ptr);
        return ret;
    }
    /**
     * Set status to encoding
     */
    start_encoding() {
        wasm.videorecorder_start_encoding(this.__wbg_ptr);
    }
    /**
     * Set status to complete
     */
    finish_encoding() {
        wasm.videorecorder_finish_encoding(this.__wbg_ptr);
    }
    /**
     * Set status to error
     */
    set_error() {
        wasm.videorecorder_set_error(this.__wbg_ptr);
    }
    /**
     * Reset recorder to idle
     */
    reset() {
        wasm.videorecorder_reset(this.__wbg_ptr);
    }
    /**
     * Generate filename based on parameters
     * Format: henon_a{a}_b{b}_eps{eps}_{animated}_{start}to{end}.mp4
     * @returns {string}
     */
    generate_filename() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.videorecorder_generate_filename(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get video config
     * @returns {VideoConfig}
     */
    get_config() {
        const ret = wasm.videorecorder_get_config(this.__wbg_ptr);
        return VideoConfig.__wrap(ret);
    }
    /**
     * Set video config
     * @param {VideoConfig} config
     */
    set_config(config) {
        _assertClass(config, VideoConfig);
        var ptr0 = config.__destroy_into_raw();
        wasm.videorecorder_set_config(this.__wbg_ptr, ptr0);
    }
    /**
     * Get expected duration in seconds based on frame count and fps
     * @returns {number}
     */
    get_expected_duration_secs() {
        const ret = wasm.videorecorder_get_expected_duration_secs(this.__wbg_ptr);
        return ret;
    }
    /**
     * Check if currently recording
     * @returns {boolean}
     */
    is_recording() {
        const ret = wasm.videorecorder_is_recording(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Check if encoding
     * @returns {boolean}
     */
    is_encoding() {
        const ret = wasm.videorecorder_is_encoding(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Get parameter overlay text for current frame
     * @param {number} current_param_value
     * @returns {string}
     */
    get_overlay_text(current_param_value) {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.videorecorder_get_overlay_text(retptr, this.__wbg_ptr, current_param_value);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);

            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);

    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };

        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg_Error_0497d5bdba9362e5 = function(arg0, arg1) {
        const ret = Error(getStringFromWasm0(arg0, arg1));
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_String_8f0eb39a4a4c2f66 = function(arg0, arg1) {
        const ret = String(getObject(arg1));
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg_buffer_a1a27a0dfa70165d = function(arg0) {
        const ret = getObject(arg0).buffer;
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_call_fbe8be8bf6436ce5 = function() { return handleError(function (arg0, arg1) {
        const ret = getObject(arg0).call(getObject(arg1));
        return addHeapObject(ret);
    }, arguments) };
    imports.wbg.__wbg_done_4d01f352bade43b7 = function(arg0) {
        const ret = getObject(arg0).done;
        return ret;
    };
    imports.wbg.__wbg_error_51ecdd39ec054205 = function(arg0) {
        console.error(getObject(arg0));
    };
    imports.wbg.__wbg_error_7534b8e9a36f1ab4 = function(arg0, arg1) {
        let deferred0_0;
        let deferred0_1;
        try {
            deferred0_0 = arg0;
            deferred0_1 = arg1;
            console.error(getStringFromWasm0(arg0, arg1));
        } finally {
            wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
        }
    };
    imports.wbg.__wbg_error_95a132f74cc5e61a = function(arg0, arg1) {
        console.error(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg_get_92470be87867c2e5 = function() { return handleError(function (arg0, arg1) {
        const ret = Reflect.get(getObject(arg0), getObject(arg1));
        return addHeapObject(ret);
    }, arguments) };
    imports.wbg.__wbg_get_a131a44bd1eb6979 = function(arg0, arg1) {
        const ret = getObject(arg0)[arg1 >>> 0];
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_getwithrefkey_1dc361bd10053bfe = function(arg0, arg1) {
        const ret = getObject(arg0)[getObject(arg1)];
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_instanceof_ArrayBuffer_a8b6f580b363f2bc = function(arg0) {
        let result;
        try {
            result = getObject(arg0) instanceof ArrayBuffer;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_Uint8Array_ca460677bc155827 = function(arg0) {
        let result;
        try {
            result = getObject(arg0) instanceof Uint8Array;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_isArray_2a07fd175d45c496 = function(arg0) {
        const ret = Array.isArray(getObject(arg0));
        return ret;
    };
    imports.wbg.__wbg_isArray_5f090bed72bd4f89 = function(arg0) {
        const ret = Array.isArray(getObject(arg0));
        return ret;
    };
    imports.wbg.__wbg_isSafeInteger_90d7c4674047d684 = function(arg0) {
        const ret = Number.isSafeInteger(getObject(arg0));
        return ret;
    };
    imports.wbg.__wbg_iterator_4068add5b2aef7a6 = function() {
        const ret = Symbol.iterator;
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_length_ab6d22b5ead75c72 = function(arg0) {
        const ret = getObject(arg0).length;
        return ret;
    };
    imports.wbg.__wbg_length_f00ec12454a5d9fd = function(arg0) {
        const ret = getObject(arg0).length;
        return ret;
    };
    imports.wbg.__wbg_log_641e3de5f4c9be3e = function(arg0, arg1) {
        console.log(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg_log_ea240990d83e374e = function(arg0) {
        console.log(getObject(arg0));
    };
    imports.wbg.__wbg_new_07b483f72211fd66 = function() {
        const ret = new Object();
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_58353953ad2097cc = function() {
        const ret = new Array();
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_8a6f238a6ece86ea = function() {
        const ret = new Error();
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_a979b4b45bd55c7f = function() {
        const ret = new Map();
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_e52b3efaaa774f96 = function(arg0) {
        const ret = new Uint8Array(getObject(arg0));
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_next_8bb824d217961b5d = function(arg0) {
        const ret = getObject(arg0).next;
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_next_e2da48d8fff7439a = function() { return handleError(function (arg0) {
        const ret = getObject(arg0).next();
        return addHeapObject(ret);
    }, arguments) };
    imports.wbg.__wbg_now_eb0821f3bd9f6529 = function() {
        const ret = Date.now();
        return ret;
    };
    imports.wbg.__wbg_set_3f1d0b984ed272ed = function(arg0, arg1, arg2) {
        getObject(arg0)[takeObject(arg1)] = takeObject(arg2);
    };
    imports.wbg.__wbg_set_7422acbe992d64ab = function(arg0, arg1, arg2) {
        getObject(arg0)[arg1 >>> 0] = takeObject(arg2);
    };
    imports.wbg.__wbg_set_d6bdfd275fb8a4ce = function(arg0, arg1, arg2) {
        const ret = getObject(arg0).set(getObject(arg1), getObject(arg2));
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_set_fe4e79d1ed3b0e9b = function(arg0, arg1, arg2) {
        getObject(arg0).set(getObject(arg1), arg2 >>> 0);
    };
    imports.wbg.__wbg_stack_0ed75d68575b0f3c = function(arg0, arg1) {
        const ret = getObject(arg1).stack;
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg_value_17b896954e14f896 = function(arg0) {
        const ret = getObject(arg0).value;
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_as_number = function(arg0) {
        const ret = +getObject(arg0);
        return ret;
    };
    imports.wbg.__wbindgen_bigint_from_i64 = function(arg0) {
        const ret = arg0;
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_bigint_from_u64 = function(arg0) {
        const ret = BigInt.asUintN(64, arg0);
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_bigint_get_as_i64 = function(arg0, arg1) {
        const v = getObject(arg1);
        const ret = typeof(v) === 'bigint' ? v : undefined;
        getDataViewMemory0().setBigInt64(arg0 + 8 * 1, isLikeNone(ret) ? BigInt(0) : ret, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
    };
    imports.wbg.__wbindgen_boolean_get = function(arg0) {
        const v = getObject(arg0);
        const ret = typeof(v) === 'boolean' ? (v ? 1 : 0) : 2;
        return ret;
    };
    imports.wbg.__wbindgen_debug_string = function(arg0, arg1) {
        const ret = debugString(getObject(arg1));
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbindgen_in = function(arg0, arg1) {
        const ret = getObject(arg0) in getObject(arg1);
        return ret;
    };
    imports.wbg.__wbindgen_is_bigint = function(arg0) {
        const ret = typeof(getObject(arg0)) === 'bigint';
        return ret;
    };
    imports.wbg.__wbindgen_is_function = function(arg0) {
        const ret = typeof(getObject(arg0)) === 'function';
        return ret;
    };
    imports.wbg.__wbindgen_is_object = function(arg0) {
        const val = getObject(arg0);
        const ret = typeof(val) === 'object' && val !== null;
        return ret;
    };
    imports.wbg.__wbindgen_is_string = function(arg0) {
        const ret = typeof(getObject(arg0)) === 'string';
        return ret;
    };
    imports.wbg.__wbindgen_is_undefined = function(arg0) {
        const ret = getObject(arg0) === undefined;
        return ret;
    };
    imports.wbg.__wbindgen_jsval_eq = function(arg0, arg1) {
        const ret = getObject(arg0) === getObject(arg1);
        return ret;
    };
    imports.wbg.__wbindgen_jsval_loose_eq = function(arg0, arg1) {
        const ret = getObject(arg0) == getObject(arg1);
        return ret;
    };
    imports.wbg.__wbindgen_memory = function() {
        const ret = wasm.memory;
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_number_get = function(arg0, arg1) {
        const obj = getObject(arg1);
        const ret = typeof(obj) === 'number' ? obj : undefined;
        getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
    };
    imports.wbg.__wbindgen_number_new = function(arg0) {
        const ret = arg0;
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_object_clone_ref = function(arg0) {
        const ret = getObject(arg0);
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_object_drop_ref = function(arg0) {
        takeObject(arg0);
    };
    imports.wbg.__wbindgen_string_get = function(arg0, arg1) {
        const obj = getObject(arg1);
        const ret = typeof(obj) === 'string' ? obj : undefined;
        var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbindgen_string_new = function(arg0, arg1) {
        const ret = getStringFromWasm0(arg0, arg1);
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_throw = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };

    return imports;
}

function __wbg_init_memory(imports, memory) {

}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedDataViewMemory0 = null;
    cachedUint8ArrayMemory0 = null;



    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();

    __wbg_init_memory(imports);

    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }

    const instance = new WebAssembly.Instance(module, imports);

    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('henon_periodic_orbits_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    __wbg_init_memory(imports);

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
