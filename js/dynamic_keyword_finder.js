import { app } from "../../scripts/app.js";

const NODE_NAME = "DynamicKeywordFinder";
const MAX_FIELDS = 20;
const STATE_KEY = "_dynamic_keyword_finder";

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function getCount(node) {
    const w = getWidget(node, "field_count");
    const n = Number(w?.value ?? 1);
    return Math.max(1, Math.min(MAX_FIELDS, Number.isFinite(n) ? Math.floor(n) : 1));
}

function getState(node) {
    node._dkfState ||= { keywords: {} };
    return node._dkfState;
}

function loadState(node) {
    const saved = node.properties?.[STATE_KEY];
    if (saved) {
        node._dkfState = { keywords: { ...(saved.keywords || {}) } };
    } else {
        getState(node);
    }
}

function saveState(node) {
    const state = getState(node);
    node.properties ||= {};

    for (let i = 1; i <= MAX_FIELDS; i++) {
        const w = getWidget(node, `keyword_${i}`);
        if (w) state.keywords[i] = String(w.value ?? "");
    }

    node.properties[STATE_KEY] = {
        keywords: { ...state.keywords },
    };
}

function removeDynamicWidgets(node) {
    if (!node.widgets) return;
    for (let i = node.widgets.length - 1; i >= 0; i--) {
        const w = node.widgets[i];
        if (w?._dkfDynamic) {
            try { w.onRemove?.(); } catch (_) {}
            node.widgets.splice(i, 1);
        }
    }
}

function removeDynamicInputs(node) {
    if (!node.inputs) return;
    for (let i = node.inputs.length - 1; i >= 0; i--) {
        const input = node.inputs[i];
        if (!input?._dkfDynamic) continue;
        if (input.link != null && node.graph) {
            try { node.graph.removeLink(input.link); } catch (_) {}
        }
        try { node.removeInput(i); } catch (_) { node.inputs.splice(i, 1); }
    }
}

function makeKeywordWidget(node, name, value) {
    const widget = node.addWidget(
        "text",
        name,
        String(value ?? ""),
        function() {
            saveState(node);
            updateOutputs(node);
        },
        { serialize: true }
    );
    widget.serialize = true;
    widget._dkfDynamic = true;
    return widget;
}

function addWidgetInput(node, widget) {
    const input = node.addInput(widget.name, "STRING", {
        widget: { name: widget.name },
    });
    input._dkfDynamic = true;
    return input;
}

function updateOutputs(node) {
    const count = getCount(node);
    if (!node.outputs) node.outputs = [];

    // Keep output index 0 = keyword_1 Found, 1 = keyword_2 Found, etc.
    // We deliberately manipulate the existing array instead of calling
    // addOutput/removeOutput, avoiding LiteGraph's automatic computeSize().
    if (node.graph) {
        for (let i = count; i < node.outputs.length; i++) {
            const out = node.outputs[i];
            if (!out?.links) continue;
            for (const linkId of [...out.links]) {
                try { node.graph.removeLink(linkId); } catch (_) {}
            }
        }
    }

    if (node.outputs.length > count) node.outputs.length = count;

    while (node.outputs.length < count) {
        const i = node.outputs.length + 1;
        node.outputs.push({
            name: `keyword_${i}_found`,
            label: `keyword_${i} Found`,
            type: "BOOLEAN",
            links: null,
        });
    }

    for (let i = 1; i <= count; i++) {
        const out = node.outputs[i - 1];
        const w = getWidget(node, `keyword_${i}`);
        const label = String(w?.value ?? "").trim() || `keyword_${i}`;
        if (out) {
            out.name = `${label}_found`;
            out.label = `${label} Found`;
            out.type = "BOOLEAN";
            out.links ??= null;
        }
    }

    node.setDirtyCanvas?.(true, true);
}

function rebuildFields(node) {
    const countWidget = getWidget(node, "field_count");
    if (!countWidget) return;

    loadState(node);
    const state = getState(node);
    const count = getCount(node);

    if (node._dkfFieldsReady && node._dkfBuiltCount === count) {
        updateOutputs(node);
        return;
    }

    saveState(node);

    if (node._dkfFieldsReady) {
        removeDynamicInputs(node);
        removeDynamicWidgets(node);
        node._dkfFieldsReady = false;
    }

    const originalSetSize = node.setSize;
    node.setSize = function() {};
    try {
        const created = [];

        for (let i = 1; i <= count; i++) {
            const keywordWidget = makeKeywordWidget(
                node,
                `keyword_${i}`,
                state.keywords[i] ?? ""
            );
            created.push(keywordWidget);
            addWidgetInput(node, keywordWidget);
        }

        // addWidget() appends. Move only the newly-created objects after the
        // fixed controls; do not clone/recreate them.
        const appendedStart = node.widgets.length - created.length;
        node.widgets.splice(appendedStart, created.length);

        const fixed = ["text", "field_count", "whole_word"];
        const positions = fixed
            .map(name => node.widgets.findIndex(w => w.name === name))
            .filter(i => i >= 0);
        const insertAt = positions.length ? Math.max(...positions) + 1 : 0;
        node.widgets.splice(insertAt, 0, ...created);

        node._dkfFieldsReady = true;
        node._dkfBuiltCount = count;
    } finally {
        node.setSize = originalSetSize;
    }

    updateOutputs(node);
    saveState(node);
    node.setDirtyCanvas?.(true, true);
}

function hookCount(node) {
    const w = getWidget(node, "field_count");
    if (!w || w._dkfCountHooked) return;

    const old = w.callback;
    w.callback = function(...args) {
        if (old) old.apply(this, args);
        saveState(node);
        rebuildFields(node);
    };
    w._dkfCountHooked = true;
}

function hookSerialization(node) {
    if (node._dkfSerializeHooked) return;

    const old = node.onSerialize;
    node.onSerialize = function(output) {
        saveState(this);
        if (old) old.call(this, output);
    };

    node._dkfSerializeHooked = true;
}

function initialize(node) {
    if (node.comfyClass !== NODE_NAME || node._dkfInitialized) return;

    node._dkfInitialized = true;
    loadState(node);
    hookSerialization(node);
    hookCount(node);
    rebuildFields(node);
}

app.registerExtension({
    name: "Comfy.DynamicKeywordFinder.v5",

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
            setTimeout(() => initialize(this), 0);
            return result;
        };
    },
});
