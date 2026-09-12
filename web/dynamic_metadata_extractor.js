import { app } from "../../scripts/app.js";

const NODE_TYPE = "DynamicMetadataExtractor";
const MAX_FIELDS = 2;

function outputName(field, found) {
    return found ? `field_${field}_found` : `field_${field}`;
}

function outputType(index) {
    return index % 2 === 1 ? "BOOLEAN" : "STRING";
}

function syncOutputs(node, count) {
    const desired = count * 2;

    // The Python node declares a 40-slot static return signature for backend
    // compatibility. The actual canvas node is trimmed to the selected count.
    while ((node.outputs?.length ?? 0) > desired) {
        node.removeOutput(node.outputs.length - 1);
    }

    while ((node.outputs?.length ?? 0) < desired) {
        const index = node.outputs?.length ?? 0;
        const field = Math.floor(index / 2) + 1;
        node.addOutput(outputName(field, index % 2 === 1), outputType(index));
    }

    for (let i = 1; i <= count; i++) {
        const valueOutput = node.outputs[(i - 1) * 2];
        const foundOutput = node.outputs[(i - 1) * 2 + 1];
        const widget = node.widgets?.find(w => w.name === `field_${i}`);
        const key = String(widget?.value ?? "").trim();

        if (valueOutput) {
            valueOutput.name = key || `field_${i}`;
            valueOutput.label = key || `field_${i}`;
        }
        if (foundOutput) {
            foundOutput.name = key ? `${key}_found` : `field_${i}_found`;
            foundOutput.label = key ? `${key}_found` : `field_${i}_found`;
        }
    }
}

function updateDynamicFields(node) {
    const countWidget = node.widgets?.find(w => w.name === "field_count");
    const count = Math.max(
        1,
        Math.min(MAX_FIELDS, Number.parseInt(countWidget?.value ?? 1, 10) || 1)
    );

    // field_count is supplied by Python as a required INT widget. Do not add a
    // second copy here; doing so creates an input the execution graph cannot
    // resolve.
    for (let i = 1; i <= MAX_FIELDS; i++) {
        const widget = node.widgets?.find(w => w.name === `field_${i}`);
        if (!widget) continue;

        widget.hidden = i > count;
        widget.label = i <= count ? `Field ${i}` : "";
    }

    syncOutputs(node, count);

    node.setSize(node.computeSize());
    node.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "DynamicMetadataExtractor",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const originalCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            originalCreated?.apply(this, arguments);

            // The server definition contains the optional field_1..field_20
            // inputs. We only hide the unused widgets; their values remain
            // serializable so changing the count does not destroy entries.
            for (let i = 1; i <= MAX_FIELDS; i++) {
                const widget = this.widgets?.find(w => w.name === `field_${i}`);
                if (widget) {
                    widget.hidden = true;
                    widget.serialize = true;
                    widget.label = `Field ${i}`;
                }
            }

            updateDynamicFields(this);
        };

        const originalConfigure = nodeType.prototype.configure;

        nodeType.prototype.configure = function (info) {
            originalConfigure?.apply(this, arguments);
            setTimeout(() => updateDynamicFields(this), 0);
        };
    },
});
