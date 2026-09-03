import { app } from "../../scripts/app.js";

const NODE_NAME = "DynamicKeywordExpander";
const MAX_FIELDS = 20;
const OUTPUT_BASE = 2;
const STATE_KEY = "_dynamic_keyword_expander";

function getCount(node) {
    const w = node.widgets?.find(x => x.name === "field_count");
    return Math.max(1, Math.min(MAX_FIELDS, Number(w?.value ?? 1)));
}

function getBooleanOnly(node) {
    const w = node.widgets?.find(x => x.name === "boolean_only");
    return w ? Boolean(w.value) : false;
}

function getState(node) {
    node._kwState ||= { keywords: {}, expansions: {} };
    return node._kwState;
}

function loadState(node) {
    const p = node.properties?.[STATE_KEY];
    if (p) {
        node._kwState = {
            keywords: { ...(p.keywords || {}) },
            expansions: { ...(p.expansions || {}) },
        };
    } else {
        getState(node);
    }
}

function saveState(node) {
    const s = getState(node);
    node.properties ||= {};

    for (let i = 1; i <= MAX_FIELDS; i++) {
        const k = node.widgets?.find(w => w.name === `keyword_${i}`);
        const e = node.widgets?.find(w => w.name === `expansion_${i}`);
        if (k) s.keywords[i] = String(k.value ?? "");
        if (e) s.expansions[i] = String(e.value ?? "");
    }

    node.properties[STATE_KEY] = {
        keywords: { ...s.keywords },
        expansions: { ...s.expansions },
    };
}

function makeTextWidget(node, name, value, multiline) {
    const widget = node.addWidget(
        "text",
        name,
        String(value ?? ""),
        function() {
            saveState(node);
            updateOutputs(node);
        },
        multiline ? { multiline: true, serialize: true } : { serialize: true }
    );

    widget.serialize = true;
    widget._kwDynamic = true;
    return widget;
}

function removeDynamicWidgets(node) {
    if (!node.widgets) return;
    for (let i = node.widgets.length - 1; i >= 0; i--) {
        const w = node.widgets[i];
        if (w._kwDynamic) {
            try { w.onRemove?.(); } catch (_) {}
            node.widgets.splice(i, 1);
        }
    }
}

function removeDynamicInputs(node) {
    if (!node.inputs) return;
    for (let i = node.inputs.length - 1; i >= 0; i--) {
        const input = node.inputs[i];
        if (input?._kwDynamic) {
            if (input.link != null && node.graph) {
                try { node.graph.removeLink(input.link); } catch (_) {}
            }
            try { node.removeInput(i); } catch (_) { node.inputs.splice(i, 1); }
        }
    }
}

function addWidgetInput(node, widget) {
    // ComfyUI widgets and input sockets coexist. Associate the socket with
    // the widget so a STRING-producing node can feed the dynamic field.
    // The backend INPUT_TYPES accepts these dynamic names.
    const input = node.addInput(widget.name, "STRING", {
        widget: { name: widget.name },
    });
    input._kwDynamic = true;
    return input;
}

function rebuildFields(node) {
    loadState(node);
    const s = getState(node);
    const count = getCount(node);
    const savedSize = node.size ? [...node.size] : null;

    // Keep this deliberately close to the proven v5 widget implementation.
    // The only major architectural change in v15 is output handling.
    if (node._kwFieldsReady && node._kwBuiltCount === count) {
        updateOutputs(node);
        if (savedSize) node.setSize([...savedSize]);
        return;
    }

    if (node._kwFieldsReady) {
        saveState(node);
        removeDynamicInputs(node);
        removeDynamicWidgets(node);
        node._kwFieldsReady = false;
    }

    const originalSetSize = node.setSize;
    node.setSize = function(_size) {};
    try {
        const created = [];
        const booleanOnly = getBooleanOnly(node);
        for (let i = 1; i <= count; i++) {
            const keywordWidget = makeTextWidget(node, `keyword_${i}`, s.keywords[i] ?? "", false);
            created.push(keywordWidget);
            addWidgetInput(node, keywordWidget);

            const expansionWidget = makeTextWidget(node, `expansion_${i}`, s.expansions[i] ?? "", true);
            expansionWidget.hidden = booleanOnly;
            created.push(expansionWidget);
            if (!booleanOnly) {
                addWidgetInput(node, expansionWidget);
            }
        }

        // addWidget appends to node.widgets. Put dynamic widgets after the
        // fixed controls without creating a second copy of them.
        let insertAt = node.widgets.length - created.length;
        const fixed = ["text", "field_count", "whole_word", "replace_all", "boolean_only"];
        const positions = fixed
            .map(name => node.widgets.findIndex(w => w.name === name))
            .filter(i => i >= 0);
        if (positions.length) insertAt = Math.max(...positions) + 1;

        // Remove the newly appended widgets, then insert the same objects once.
        node.widgets.splice(node.widgets.length - created.length, created.length);
        node.widgets.splice(insertAt, 0, ...created);

        node._kwFieldsReady = true;
        node._kwBuiltCount = count;
    } finally {
        node.setSize = originalSetSize;
    }

    updateOutputs(node);
    saveState(node);
    if (savedSize) node.setSize([...savedSize]);
    node.setDirtyCanvas(true, true);
}

function makeOutput(name, type, label) {
    return {
        name,
        type,
        links: null,
        label,
    };
}

function disconnectRemovedOutputs(node, wanted) {
    if (!node.outputs || !node.graph) return;
    for (let i = wanted; i < node.outputs.length; i++) {
        const out = node.outputs[i];
        if (!out?.links) continue;
        for (const linkId of [...out.links]) {
            try { node.graph.removeLink(linkId); } catch (_) {}
        }
    }
}

function updateOutputs(node) {
    const count = getCount(node);
    const booleanOnly = getBooleanOnly(node);
    // Keep the live output slots aligned with the backend return ordering.
    // Even in boolean-only mode we must retain the hidden STRING slot before
    // each BOOLEAN Found slot, because output index determines the backend
    // type during connection validation. We hide the Text sockets rather
    // than compacting them away.
    const wanted = OUTPUT_BASE + count * 2;
    const savedSize = node.size ? [...node.size] : null;

    if (!node.outputs) node.outputs = [];

    // The backend advertises the complete possible output type surface.
    // The live frontend surface is reduced to the active fields/mode.  Unused sockets
    // are removed from the live node, so they do not contribute to layout or
    // node minimum size.  Native sockets retain their real BOOLEAN/STRING
    // types for connection validation.
    const originalSetSize = node.setSize;
    node.setSize = function(_size) {};
    try {
        while (node.outputs.length > wanted) {
            const slot = node.outputs.length - 1;
            try { node.removeOutput(slot); } catch (_) {
                node.outputs.splice(slot, 1);
            }
        }

        while (node.outputs.length < wanted) {
            const dynamicIndex = node.outputs.length - OUTPUT_BASE;
            const field = Math.floor(dynamicIndex / 2) + 1;
            const isFound = dynamicIndex % 2 === 1;
            node.addOutput(
                `keyword_${field}_${isFound ? "found" : "text"}`,
                isFound ? "BOOLEAN" : "STRING",
                { label: `keyword_${field} ${isFound ? "Found" : "Text"}` }
            );
        }

        for (let i = 1; i <= count; i++) {
            const k = node.widgets?.find(w => w.name === `keyword_${i}`);
            const label = String(k?.value ?? "").trim() || `keyword_${i}`;

            const textOut = node.outputs[OUTPUT_BASE + (i - 1) * 2];
            const foundOut = node.outputs[OUTPUT_BASE + (i - 1) * 2 + 1];

            if (textOut) {
                textOut.name = `${label}_text`;
                textOut.label = `${label} Text`;
                textOut.type = "STRING";
                textOut.hidden = booleanOnly;
            }
            if (foundOut) {
                foundOut.name = `${label}_found`;
                foundOut.label = `${label} Found`;
                foundOut.type = "BOOLEAN";
                foundOut.hidden = false;
            }
        }
    } finally {
        node.setSize = originalSetSize;
    }

    // Now that only active outputs remain, let LiteGraph calculate the true
    // minimum size. Preserve a previously saved/larger size, but never use the
    // 20-field definition as the minimum.
    const minSize = node.computeSize ? node.computeSize() : [230, 80];
    const baseSize = savedSize || [230, 80];
    originalSetSize.call(node, [
        Math.max(baseSize[0], minSize[0]),
        Math.max(baseSize[1], minSize[1]),
    ]);

    node.setDirtyCanvas(true, true);
}

function hookCount(node) {
    const w = node.widgets?.find(x => x.name === "field_count");
    if (!w || w._kwCountHooked) return;

    const old = w.callback;
    w.callback = function(...args) {
        if (old) old.apply(this, args);
        saveState(node);
        rebuildFields(node);
    };
    w._kwCountHooked = true;
}

function hookBooleanOnlyToggle(node) {
    const w = node.widgets?.find(x => x.name === "boolean_only");
    if (!w || w._kwBooleanOnlyHooked) return;

    const old = w.callback;
    w.callback = function(...args) {
        if (old) old.apply(this, args);
        saveState(node);

        // Toggle visibility of expansion widgets. Inputs are rebuilt so the
        // expansion sockets disappear in boolean-only mode as well.
        const originalSetSize = node.setSize;
        node.setSize = function(_size) {};
        try {
            removeDynamicInputs(node);
            const booleanOnly = getBooleanOnly(node);
            for (const widget of node.widgets || []) {
                if (!widget._kwDynamic) continue;
                if (/^keyword_\d+$/.test(widget.name)) {
                    widget.hidden = false;
                    addWidgetInput(node, widget);
                } else if (/^expansion_\d+$/.test(widget.name)) {
                    widget.hidden = booleanOnly;
                    if (!booleanOnly) addWidgetInput(node, widget);
                }
            }
        } finally {
            node.setSize = originalSetSize;
        }

        updateOutputs(node);
        node.setDirtyCanvas(true, true);
    };
    w._kwBooleanOnlyHooked = true;
}

function hookSerialization(node) {
    if (node._kwSerializeHooked) return;
    const oldOnSerialize = node.onSerialize;
    node.onSerialize = function(output) {
        saveState(this);
        if (oldOnSerialize) oldOnSerialize.call(this, output);
        this.properties ||= {};
        this.properties[STATE_KEY] = {
            keywords: { ...getState(this).keywords },
            expansions: { ...getState(this).expansions },
        };
    };
    node._kwSerializeHooked = true;
}

function initialize(node) {
    if (node.comfyClass !== NODE_NAME || node._kwInitialized) return;
    node._kwInitialized = true;
    hookSerialization(node);
    loadState(node);
    hookCount(node);
    hookBooleanOnlyToggle(node);
    rebuildFields(node);
}

app.registerExtension({
    name: "Comfy.DynamicKeywordExpander.v25",

    async nodeCreated(node) {
        if (node.comfyClass !== NODE_NAME) return;
        setTimeout(() => initialize(node), 0);
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const oldConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function(info) {
            const result = oldConfigure
                ? oldConfigure.apply(this, arguments)
                : undefined;

            // ComfyUI has already restored the serialized outputs and size.
            // Rebuild widgets, but never use addOutput/removeOutput for the
            // dynamic output surface.
            setTimeout(() => initialize(this), 0);
            return result;
        };
    },
});
