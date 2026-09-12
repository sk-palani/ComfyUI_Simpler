import { app } from "../../scripts/app.js";

const LOG_PREFIX = "[DynamicKeywordExpander]";

// Must match MAX_FIELDS in nodes.py
const MAX_FIELDS = 20;
const NODE_NAME = "DynamicKeywordExpander";
// Number of always-visible, non-dynamic widgets declared before the
// keyword_i/expansion_i pairs in INPUT_TYPES: text, field_count,
// whole_word, replace_all.
const FIXED_WIDGET_COUNT = 4;

console.log(`${LOG_PREFIX} extension script loaded`);

function isOurNode(node) {
    return node?.comfyClass === NODE_NAME || node?.type === NODE_NAME;
}

function clampCount(value) {
    return Math.max(1, Math.min(MAX_FIELDS, Math.round(Number(value) || 1)));
}

// Replace the full contents of an array IN PLACE (same reference kept)
// rather than reassigning `node.widgets = newArray`, for compatibility with
// the Vue-based ComfyUI_frontend's reactivity around the canvas.
function replaceContentsInPlace(arr, items) {
    arr.length = 0;
    for (const item of items) arr.push(item);
}

function applyFieldCount(node, rawCount) {
    if (!Array.isArray(node.widgets)) {
        console.warn(`${LOG_PREFIX} node.widgets not ready yet, skipping`);
        return;
    }

    // Capture the node's full, unmodified widget list exactly once, as an
    // independent copy (not a reference into node.widgets, since that gets
    // mutated in place below).
    if (!node._allWidgetsBackup) {
        node._allWidgetsBackup = node.widgets.slice();
    }

    const count = clampCount(rawCount);
    node.properties = node.properties || {};
    node.properties.field_count = count;

    const fixedWidgets = node._allWidgetsBackup.slice(0, FIXED_WIDGET_COUNT);
    const dynamicWidgets = node._allWidgetsBackup.slice(FIXED_WIDGET_COUNT);
    const desiredWidgets = fixedWidgets.concat(dynamicWidgets.slice(0, count * 2));
    replaceContentsInPlace(node.widgets, desiredWidgets);

    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
    node.graph?.setDirtyCanvas?.(true, true);

    console.log(`${LOG_PREFIX} applied field_count=${count} -> widgets=${node.widgets.length}`);
}

function initNode(node) {
    try {
        const fcWidget = node.widgets?.find((w) => w.name === "field_count");
        const initialCount = node.properties?.field_count ?? fcWidget?.value ?? 1;

        if (fcWidget && !fcWidget._dkeWrapped) {
            const origCallback = fcWidget.callback;
            fcWidget.callback = (value, ...rest) => {
                const r = origCallback?.call(fcWidget, value, ...rest);
                applyFieldCount(node, fcWidget.value);
                return r;
            };
            fcWidget._dkeWrapped = true;
        }

        applyFieldCount(node, initialCount);
        node._dkeAppliedOnce = true;
    } catch (err) {
        console.error(`${LOG_PREFIX} failed to initialize node`, err);
    }
}

app.registerExtension({
    name: "comfyui.DynamicKeywordExpander",

    // Called right after ComfyUI constructs a node -- for BOTH a brand-new
    // node AND a node being loaded from a saved workflow. Critically, for
    // the loaded-workflow case this fires *before* node.configure() runs,
    // and configure() restores saved widget values by matching them to
    // node.widgets *by array index*. If we truncate node.widgets here
    // immediately, configure() has nowhere to put the saved values for the
    // rows beyond the (still-default) field_count=1, and they're silently
    // lost -- which is exactly the reload corruption bug this fixes.
    //
    // Deferring via requestAnimationFrame lets configure() (synchronous,
    // same tick) finish first. For a genuinely new node, this just delays
    // the initial collapse-to-1-row by a frame (a harmless, brief flash).
    nodeCreated(node) {
        if (!isOurNode(node)) return;
        console.log(`${LOG_PREFIX} nodeCreated (deferring initial apply)`, node);
        requestAnimationFrame(() => {
            if (!node._dkeAppliedOnce) {
                initNode(node);
            }
        });
    },

    // Called for every node after a saved workflow finishes loading --
    // node.configure() has already run, so node.properties and widget
    // values are the fully-restored saved state. Safe to read/truncate
    // immediately here (this runs before the deferred nodeCreated callback
    // above gets its turn, so it wins the race and does the real work).
    loadedGraphNode(node) {
        if (!isOurNode(node)) return;
        console.log(`${LOG_PREFIX} loadedGraphNode`, node);
        // Fresh instance: drop any stale backup so it gets recaptured from
        // the just-restored full widget list/values.
        node._allWidgetsBackup = null;
        initNode(node);
    },
});